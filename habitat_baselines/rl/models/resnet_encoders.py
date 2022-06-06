import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from gym import spaces
from habitat import logger
from habitat.core.utils import try_cv2_import
from habitat_baselines.rl.models.rednet import RedNet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from PIL import Image
import pickle
from torchvision.models.feature_extraction import  create_feature_extractor

cv2 = try_cv2_import()

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class SemanticObjectDetector(nn.Module):
    def __init__(self, device, obs_size, extract_layer=3):
        super().__init__()
        
        self.device = device
        self.obs_size = obs_size
        self.extract_layer = extract_layer
        self.in_channels = 1024
        
        self.model = self.get_object_detection_model()
        # disable gradients for resnet, params frozen
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.model.to('cpu')
        #self.model.to(device)
        
        self.feature_extractor = self.get_feature_extractor()
        # disable gradients for resnet, params frozen
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        #self.feature_extractor.to(device)
        self.feature_extractor.to('cpu')
        
        self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=128,
                    kernel_size=7, 
                    stride=1, 
                    padding=3, 
                    bias=False,
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=128,
                    out_channels=32,
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False,
                ),
            )
        #self.cnn.to(device)
        
        self.tranform = torchvision.transforms.ToTensor()
        
    def get_feature_extractor(self):
        feature_extractor = create_feature_extractor(self.model.backbone.body, 
                                                     {'layer1': 'feat1', 'layer2': 'feat2',
                                                      'layer3': 'feat3', 'layer4': 'feat4'})
        
        return feature_extractor

    def get_object_detection_model(self, num_classes=9):
        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load('data/pretrained_models/obj_det_real.ckpt'))
        return model

    def apply_nms(self, orig_prediction, iou_thresh=0.3):
        # torchvision returns the indices of the bboxes to keep
        keep = torchvision.ops.nms(orig_prediction['boxes'],
                                   orig_prediction['scores'], iou_thresh)

        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'][keep]
        final_prediction['scores'] = final_prediction['scores'][keep]
        final_prediction['labels'] = final_prediction['labels'][keep]

        return final_prediction

    def filter_pred(self, pred, img, threshold=0.95, threshold_knn=0.75):
        pred['boxes'] = pred['boxes'].detach().cpu().numpy().tolist()
        pred['scores'] = pred['scores'].detach().cpu().numpy().tolist()
        pred['labels'] = pred['labels'].detach().cpu().numpy().tolist()
        res = {}
        res['boxes'] = []
        res['scores'] = []
        res['labels'] = []
        colors = []
        for idx, score in enumerate(pred['scores']):
            if score > threshold and pred['labels'][idx] != 0:
                box = pred['boxes'][idx]
                res['boxes'].append(box)
                res['scores'].append(pred['scores'][idx])
                res['labels'].append(pred['labels'][idx])
                #center_x = int((box[2] + box[0]) / 2)
                #center_y = int((box[3] + box[1]) / 2)
                #colors.append(img[center_y][center_x])

        return res

    def predict(self, img):
        img_trans = self.tranform(Image.fromarray(img[0,:,:,:].cpu().numpy().astype(np.uint8)))
        prediction = self.model([img_trans.to('cuda')])[0]
        nms_prediction = self.apply_nms(prediction, iou_thresh=0.2)
        res = self.filter_pred(nms_prediction, img)
        return res
    
    def extract_features(self, img):
        feats = self.feature_extractor(img)
        return feats[f'feat{self.extract_layer}']
    
    def forward(self, observations):
        img = observations['rgb'].permute(0,3,1,2).type(torch.float)
        feats = self.extract_features(img)
        cnn_feats = self.cnn(feats)
        return cnn_feats
    
    @property
    def _output_size(self):
        s = (self.cnn(self.feature_extractor(torch.rand(self.obs_size).unsqueeze(0).permute(0,3,1,2))
             [f'feat{self.extract_layer}']).data.shape)
        return s
        

class VisualRednetEncoder(nn.Module):
    r"""Based on RedNet

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, device):
        super().__init__()
        self.device = device
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # -- Create RedNet model
        cfg_rednet = {
            'arch': 'rednet',
            'resnet_pretrained': False,
            'finetune': False,
            'SUNRGBD_pretrained_weights': '',
            'n_classes': 13,
            'upsample_prediction': True,
            'load_model': 'data/rednet_mp3d_best_model.pkl',
        }

        self.model_rednet = RedNet(cfg_rednet)
        self.model_rednet = self.model_rednet.to(device)

        #print('Loading pre-trained weights: ', cfg_rednet['load_model'])
        state = torch.load(cfg_rednet['load_model'])
        model_state = state['model_state']
        model_state = self.rename_weights(model_state, 'cpu')
        self.model_rednet.load_state_dict(model_state)
        
        # disable gradients for resnet, params frozen
        for param in self.model_rednet.parameters():
            param.requires_grad = False
        self.model_rednet.eval()
        
        self.model_rednet.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.model_rednet.inplanes,
                    out_channels=32,
                    kernel_size=8,
                    stride=5,
                ),
                nn.ReLU(True),
            )

    def rename_weights(self, weights, device):
        names = list(weights.keys())
        is_module = names[0].split('.')[0] == 'module'
        if device == 'cuda' and not is_module:
            new_weights = {'module.'+k:v for k,v in weights.items()}
        elif device == 'cpu' and is_module:
            new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items()}
        else:
            new_weights = weights
        return new_weights

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

        output = self.model_rednet(rgb_observations, depth_observations)
        feats = self.model_rednet.cnn(output)
        return feats

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="NONE",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, observation_space, output_size, device, spatial_output: bool = False, flat: bool = True
    ):
        super().__init__()
        self.device = device
        self.flat = flat
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            obs_size_0 = observation_space.spaces["rgb"].shape[0]
            obs_size_1 = observation_space.spaces["rgb"].shape[1]
            if obs_size_0 != 224 or obs_size_1 != 224:
                logger.warn(
                    f"WARNING: TorchVisionResNet50: observation size {obs_size_0} is not conformant to expected ResNet input size [3x224x224]"
                )
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)

            obs_size = rgb_observations.shape[2]
            if obs_size != 224:
                # resizing rgb input image
                rgb_observations = F.interpolate(
                    rgb_observations, size=(224, 224)
                )
            
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            if self.flat:
                return self.activation(
                    self.fc(torch.flatten(resnet_output, 1))
                )  # [BATCH x OUTPUT_DIM]

            return resnet_output

class TorchVisionResNet50FeatureMap(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self, observation_space, output_size, device, global_map_depth=32
    ):
        super().__init__()
        self.device = device
        self.resnet_layer_size = 2048  # for resnet layer4
        #self.resnet_layer_size = 1024   # for resnet layer3
        linear_layer_input_size = 0
        self.global_map_depth = global_map_depth
        
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            obs_size_0 = observation_space.spaces["rgb"].shape[0]
            obs_size_1 = observation_space.spaces["rgb"].shape[1]
            if obs_size_0 != 224 or obs_size_1 != 224:
                logger.warn(
                    f"WARNING: TorchVisionResNet50: observation size {obs_size_0} is not conformant to expected ResNet input size [3x224x224]"
                )
            linear_layer_input_size += self.resnet_layer_size
        else:
            self._n_input_rgb = 0

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        #self.layer_extract = self.cnn._modules.get("avgpool")
        self.layer_extract = self.cnn._modules.get("layer4")
        #self.layer_extract = self.cnn._modules.get("layer3")

        self.rgb_encoder_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.resnet_layer_size,
                out_channels=self.global_map_depth,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
        )

        self.output_shape = (self.global_map_depth, 7, 7)

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    @property
    def output_size(self):
        return self.global_map_depth

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_projection_features" in observations:
            resnet_output = observations["rgb_projection_features"]
            return resnet_output
        else:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)

            obs_size = rgb_observations.shape[2]
            if obs_size != 224:
                # resizing rgb input image
                rgb_observations = F.interpolate(
                    rgb_observations, size=(224, 224)
                )
            
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())   # (bs, 2048, 7, 7)

            _resnet_output = self.rgb_encoder_cnn(resnet_output)

            return _resnet_output
