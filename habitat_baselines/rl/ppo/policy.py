#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from habitat_baselines.common.utils import CategoricalNet, Flatten, to_grid
from habitat_baselines.rl.models.projection import Projection, RotateTensor, get_grid
from habitat_baselines.rl.models.rnn_state_encoder import RNNStateEncoder
from habitat_baselines.rl.models.simple_cnn import (
    RGBCNNNonOracle, RGBCNNOracle, MapCNN, SimpleCNN, 
    SemanticMapCNN, SemmapDecoder, SimpleRgbCNN)
from habitat_baselines.rl.models.resnet_encoders import VisualRednetEncoder, SemanticObjectDetector
from habitat_baselines.rl.models.projection import Projection
from habitat.utils.visualizations import maps
from PIL import Image

from habitat.core.utils import try_cv2_import
cv2 = try_cv2_import()

class PolicyObjectRecog(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    @property
    def should_load_agent_state(self):
        return True

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states, obj_prob = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        loss = nn.CrossEntropyLoss()
        obj_recog_loss = loss(obj_prob, observations['objectCat'].long())
        return value, action_log_probs, distribution_entropy, rnn_hidden_states, obj_recog_loss


class PolicyNonOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, global_map

    def get_value(self, observations, rnn_hidden_states, global_map, prev_actions, masks):
        features, _, _ = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        features, rnn_hidden_states, global_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks, ev=1
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

class PolicySemantic(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    @property
    def should_load_agent_state(self):
        return True
    
    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        global_map,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, global_map, sem_goal_map, next_goal_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return (value, action, action_log_probs, rnn_hidden_states, 
                global_map)

    def get_value(self, observations, rnn_hidden_states, global_map, prev_actions, masks):
        features, _, _, _, _ = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, global_map, prev_actions, masks, action
    ):
        features, rnn_hidden_states, global_map, sem_goal_map, next_goal_map = self.net(
            observations, rnn_hidden_states, global_map, prev_actions, masks, ev=1
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        loss = nn.CrossEntropyLoss(reduction='mean')
        semantic_map_loss = loss(sem_goal_map, observations['semMap'].long())
        
        # oracle_next_goal_map = (observations['semMap'].long() == torch.add(observations['multiobjectgoal'],1).unsqueeze(1)).float()
        # #next_goal_loss = nn.BCELoss(reduction='mean')
        # next_goal_loss = nn.MSELoss(reduction='mean')
        # next_goal_map_loss = next_goal_loss(next_goal_map, oracle_next_goal_map)
        
        return (value, action_log_probs, distribution_entropy, rnn_hidden_states, 
                semantic_map_loss, None)


class PolicyOracle(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    @property
    def should_load_agent_state(self):
        return True

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )

        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states



class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)



class BaselinePolicyObjectRecog(PolicyObjectRecog):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetObjectRecog(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
            ),
            action_space.n,
        )


class BaselinePolicyNonOracle(PolicyNonOracle):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        egocentric_map_size,
        global_map_size,
        global_map_depth,
        coordinate_min,
        coordinate_max,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetNonOracle(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                egocentric_map_size=egocentric_map_size,
                global_map_size=global_map_size,
                global_map_depth=global_map_depth,
                coordinate_min=coordinate_min,
                coordinate_max=coordinate_max,
            ),
            action_space.n,
        )

class BaselinePolicySemantic(PolicySemantic):
    def __init__(
        self,
        batch_size,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        config,
        hidden_size=512,
    ):
        super().__init__(
            BaselineNetSemantic(
                batch_size,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                config=config,
            ),
            action_space.n,
        )

class BaselinePolicyOracleMap(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        hidden_size=512,
        config=None
    ):
        super().__init__(
            BaselineNetOracleMap(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                config=config
            ),
            action_space.n,
        )
        
class BaselinePolicyOracle(PolicyOracle):
    def __init__(
        self,
        agent_type,
        observation_space,
        action_space,
        goal_sensor_uuid,
        device,
        object_category_embedding_size,
        previous_action_embedding_size,
        use_previous_action,
        hidden_size=512,
        config=None
    ):
        super().__init__(
            BaselineNetOracle(
                agent_type,
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                device=device,
                object_category_embedding_size=object_category_embedding_size,
                previous_action_embedding_size=previous_action_embedding_size,
                use_previous_action=use_previous_action,
                config=config
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, global_map, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class BaselineNetNonOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, batch_size, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action,
        egocentric_map_size, global_map_size, global_map_depth, coordinate_min, coordinate_max
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.egocentric_map_size = egocentric_map_size
        self.global_map_size = global_map_size
        self.global_map_depth = global_map_depth

        self.visual_encoder = RGBCNNNonOracle(observation_space, hidden_size)
        self.map_encoder = MapCNN(51, 256, "non-oracle")        

        self.projection = Projection(egocentric_map_size, global_map_size, 
            device, coordinate_min, coordinate_max
        )

        self.to_grid = to_grid(global_map_size, coordinate_min, coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(32 * 28 * 28, 512)

        self.flatten = Flatten()

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size + 256 + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        self.full_global_map = torch.zeros(
            batch_size,
            global_map_size,
            global_map_size,
            global_map_depth,
            device=self.device,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, global_map, prev_actions, masks, ev=0):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)
        
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
        # interpolated_perception_embed = F.interpolate(perception_embed, scale_factor=256./28., mode='bilinear')
        projection = self.projection.forward(perception_embed, observations['depth'] * 10, -(observations["compass"]))
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        # grid_x_coord, grid_y_coord = grid_x.type(torch.uint8), grid_y.type(torch.uint8)
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            self.full_global_map[:bs, :, :, :] = self.full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = self.full_global_map[bs:, :, :, :] * 0
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
            else:
                agent_view = torch.FloatTensor(bs, self.global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)
            agent_view[:, :, 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            st_pose = torch.cat(
                [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                 -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                 observations['compass']], 
                 dim=1
            )
            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)
            self.full_global_map[:bs, :, :, :] = torch.max(self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1))
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    torch.zeros_like(observations['compass'])
                    ],
                    dim=1
                )
            _, trans_mat_retrieval = get_grid(st_pose_retrieval, agent_view.size(), self.device)
            translated_retrieval = F.grid_sample(self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2), trans_mat_retrieval)
            translated_retrieval = translated_retrieval[:,:,
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2), 
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2)
            ]
            final_retrieval = self.rotate_tensor.forward(translated_retrieval, observations["compass"])

            global_map_embed = self.map_encoder(final_retrieval.permute(0, 2, 3, 1))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1)
        else: 
            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)  ##verify
            with torch.cuda.device(self.device):
                agent_view = torch.cuda.FloatTensor(bs, self.global_map_depth, 51, 51).fill_(0)
            agent_view[:, :, 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2), 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            
            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            global_map_embed = self.map_encoder(final_retrieval)

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, global_map_embed, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
            return x, rnn_hidden_states, final_retrieval.permute(0, 2, 3, 1) 

class BaselineNetSemantic(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, batch_size, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action, config
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action
        self.config = config
        self.egocentric_map_size = self.config.RL.MAPS.egocentric_map_size
        self.global_map_size = self.config.RL.MAPS.global_map_size
        self.global_map_depth = self.config.RL.MAPS.global_map_depth
        self.coordinate_max = self.config.RL.MAPS.coordinate_max
        self.coordinate_min = self.config.RL.MAPS.coordinate_min
        self.local_map_size = self.config.RL.MAPS.local_map_size
        self.num_classes = self.config.RL.MAPS.num_classes + 1 # Add 1 to indicate no object when it is zero
        self.num_occ_classes = self.config.RL.MAPS.num_occ_classes
        self.linear_out = self.config.RL.MAPS.linear_out
        self.use_occupancy = self.config.RL.MAPS.USE_OCCUPANCY
        self.visual_encoder_type = self.config.RL.MAPS.visual_encoder_type

        if self.visual_encoder_type == "fasterrcnn":
            self.visual_encoder = SemanticObjectDetector(device, 
                                    obs_size=observation_space.spaces['rgb'].shape)  # Faster RCNN - finetuned on goal objects
        elif self.visual_encoder_type == "simple":
            self.visual_encoder = SimpleRgbCNN(observation_space, hidden_size) # 3CONV
        elif self.visual_encoder_type == "rednet":
            self.visual_encoder = VisualRednetEncoder(observation_space, hidden_size, device)  # RedNet
            
        self.visual_encoder_output_size = self.visual_encoder._output_size
        
        # Map projection
        self.projection = Projection(self.egocentric_map_size, self.global_map_size, 
            device, self.coordinate_min, self.coordinate_max
        )
        self.to_grid = to_grid(self.global_map_size, self.coordinate_min, self.coordinate_max)
        self.rotate_tensor = RotateTensor(device)

        self.image_features_linear = nn.Linear(
            np.prod(self.visual_encoder_output_size), 
            self.linear_out)

        self.flatten = Flatten()

        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)
        
        self.full_global_map_depth = self.visual_encoder_output_size[1]
        self.full_global_map = torch.zeros(
            batch_size,
            self.global_map_size,
            self.global_map_size,
            self.full_global_map_depth,
            device=self.device,
        )
        
        self.sem_map_decoder = SemmapDecoder(feat_dim=self.full_global_map_depth, n_obj_classes=self.num_occ_classes)

        self.object_embedding = nn.Embedding(self.num_classes, object_category_embedding_size)
        
        self.map_linear = nn.Linear(
            (self.local_map_size*self.local_map_size*self.num_occ_classes), 
            self.linear_out)
        
        self.state_encoder = RNNStateEncoder(
            (self._hidden_size           # visual_encoder
            + object_category_embedding_size    # goal_embedding
            + self.linear_out  # next goal map embedding
            + (previous_action_embedding_size 
                if self.use_previous_action else 0)    # action_embedding
            ), 
            self._hidden_size,
        )
        
        self.print_v = False # True for debugging
        self.count = 0  # used in printing images in debug mode
        
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, global_map, prev_actions, masks, ev=0):
        target_encoding = self.get_target_encoding(observations)
        goal_embed = self.object_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)
        
        perception_embed = self.visual_encoder(observations)
        # interpolated_perception_embed = F.interpolate(perception_embed, scale_factor=256./28., mode='bilinear')
        projection = self.projection.forward(perception_embed, observations['depth'] * 10, -(observations["compass"]))
        perception_embed = self.image_features_linear(self.flatten(perception_embed))
        grid_x, grid_y = self.to_grid.get_grid_coords(observations['gps'])
        # grid_x_coord, grid_y_coord = grid_x.type(torch.uint8), grid_y.type(torch.uint8)
        bs = global_map.shape[0]
        ##forward pass specific
        if ev == 0:
            self.full_global_map[:bs, :, :, :] = self.full_global_map[:bs, :, :, :] * masks.unsqueeze(1).unsqueeze(1)
            if bs != 18:
                self.full_global_map[bs:, :, :, :] = self.full_global_map[bs:, :, :, :] * 0
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    agent_view = torch.cuda.FloatTensor(bs, self.full_global_map_depth, self.global_map_size, self.global_map_size).fill_(0)
            else:
                agent_view = torch.FloatTensor(bs, self.full_global_map_depth, self.global_map_size, self.global_map_size).to(self.device).fill_(0)
            agent_view[:, :, 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2), 
                self.global_map_size//2 - math.floor(self.egocentric_map_size/2):self.global_map_size//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            st_pose = torch.cat(
                [-(grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                 -(grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2), 
                 observations['compass']], 
                 dim=1
            )
            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)
            rotated = F.grid_sample(agent_view, rot_mat)
            translated = F.grid_sample(rotated, trans_mat)
            self.full_global_map[:bs, :, :, :] = torch.max(self.full_global_map[:bs, :, :, :], translated.permute(0, 2, 3, 1))
            st_pose_retrieval = torch.cat(
                [
                    (grid_y.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    (grid_x.unsqueeze(1)-(self.global_map_size//2))/(self.global_map_size//2),
                    torch.zeros_like(observations['compass'])
                    ],
                    dim=1
                )
            _, trans_mat_retrieval = get_grid(st_pose_retrieval, agent_view.size(), self.device)
            translated_retrieval = F.grid_sample(self.full_global_map[:bs, :, :, :].permute(0, 3, 1, 2), trans_mat_retrieval)
            translated_retrieval = translated_retrieval[:,:,
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2), 
                self.global_map_size//2-math.floor(51/2):self.global_map_size//2+math.ceil(51/2)
            ]
            final_retrieval = self.rotate_tensor.forward(translated_retrieval, observations["compass"])

            # semantic map decoder
            sem_goal_map_feats, sem_goal_map = self.sem_map_decoder(final_retrieval)

            # linearize
            map_linear = self.map_linear(self.flatten(sem_goal_map))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, map_linear, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states.permute(1, 0, 2), masks)
            return x, rnn_hidden_states.permute(1, 0, 2), final_retrieval.permute(0, 2, 3, 1), sem_goal_map, None
        else: 
            global_map = global_map * masks.unsqueeze(1).unsqueeze(1)  ##verify
            with torch.cuda.device(self.device):
                agent_view = torch.cuda.FloatTensor(bs, self.full_global_map_depth, 51, 51).fill_(0)
            agent_view[:, :, 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2), 
                51//2 - math.floor(self.egocentric_map_size/2):51//2 + math.ceil(self.egocentric_map_size/2)
            ] = projection
            
            final_retrieval = torch.max(global_map, agent_view.permute(0, 2, 3, 1))

            # semantic map decoder
            sem_goal_map_feats, sem_goal_map = self.sem_map_decoder(final_retrieval.permute(0,3,1,2))

            # linearize
            map_linear = self.map_linear(self.flatten(sem_goal_map))

            if self.use_previous_action:
                action_embedding = self.action_embedding(prev_actions).squeeze(1)

            x = torch.cat((perception_embed, map_linear, goal_embed, action_embedding), dim = 1)
            x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states.permute(1, 0, 2), masks)
            return x, rnn_hidden_states.permute(1, 0, 2), final_retrieval, sem_goal_map, None
            
class BaselineNetOracle(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, agent_type, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action, config
    ):
        super().__init__()
        self.config = config
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        if agent_type == "oracle":
            _n_input_map = 32
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            if self.config.TASK_CONFIG.TASK.INCLUDE_DISTRACTORS and self.config.RL.MAPS.INCLUDE_DISTRACTOR_EMBED:
                self.distractor_embedding = nn.Embedding(9, 16)
                _n_input_map += 16
            
            self.map_encoder = MapCNN(50, 256, agent_type, _n_input_map)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            _n_input_map = 16
            if self.config.RL.MAPS.USE_OCC_IN_ORACLE_EGO:
                self.occupancy_embedding = nn.Embedding(4, 16)
                _n_input_map += 16
            self.map_encoder = MapCNN(50, 256, agent_type, _n_input_map)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
            
        
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        if self.use_previous_action:
            """ self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            ) """
            self.state_encoder = RNNStateEncoder(
                832, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        if self.agent_type != "no-map":
            global_map_embedding = []
            global_map = observations['semMap']
            
            if self.agent_type == "oracle" or (self.agent_type == "oracle-ego" and self.config.RL.MAPS.USE_OCC_IN_ORACLE_EGO):
                global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
                
            if self.config.RL.MAPS.NEXT_GOAL_IND:
                # Derive map with next goal from goal category and oracle map with goals and/or distractors
                obj_cat_map = global_map[:,:,:,1]
                distrIndexOffset = 1 if self.agent_type == "oracle" else 2 # to match categories in oracle map
                goal_map = (obj_cat_map == (target_encoding + distrIndexOffset).unsqueeze(1)).type(torch.LongTensor)
                global_map_embedding.append(self.object_embedding(goal_map.to(self.device).view(-1)).view(bs, 50, 50, -1))
            else:
                global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            
            if self.config.TASK_CONFIG.TASK.INCLUDE_DISTRACTORS and self.config.RL.MAPS.INCLUDE_DISTRACTOR_EMBED:
                global_map_embedding.append(self.distractor_embedding(global_map[:, :, :, 2].type(torch.LongTensor)
                                                                        .to(self.device).view(-1)).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)
        return x, rnn_hidden_states  

class BaselineNetOracleMap(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, agent_type, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action, config
    ):
        super().__init__()
        self.config = config
        self.agent_type = agent_type
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        if agent_type == "oracle":
            _n_input_map = 32 #48
            self.occupancy_embedding = nn.Embedding(3, 16)
            self.object_embedding = nn.Embedding(9, 16)
            self.curr_goal_embedding = nn.Embedding(2, 16)
            if self.config.RL.MAPS.ALL_GOAL_IND:
                _n_input_map += 16
            if self.config.TASK_CONFIG.TASK.INCLUDE_DISTRACTORS and self.config.RL.MAPS.INCLUDE_DISTRACTOR_EMBED:
                self.distractor_embedding = nn.Embedding(9, 16)
                _n_input_map += 16
            
            self.map_encoder = MapCNN(50, 256, agent_type, _n_input_map)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        elif agent_type == "no-map":
            self.goal_embedding = nn.Embedding(8, object_category_embedding_size)
        elif agent_type == "oracle-ego":
            _n_input_map = 16
            if self.config.RL.MAPS.USE_OCC_IN_ORACLE_EGO:
                self.occupancy_embedding = nn.Embedding(4, 16)
                _n_input_map += 16
            self.map_encoder = MapCNN(50, 256, agent_type, _n_input_map)
            self.object_embedding = nn.Embedding(10, 16)
            self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
            
        
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        if self.use_previous_action:
            """ self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            ) """
            self.state_encoder = RNNStateEncoder(
                832, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x

        if self.agent_type != "no-map":
            global_map_embedding = []
            global_map = observations['semMap']
            
            if self.agent_type == "oracle" or (self.agent_type == "oracle-ego" and self.config.RL.MAPS.USE_OCC_IN_ORACLE_EGO):
                global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
                
            if self.config.RL.MAPS.NEXT_GOAL_IND:
                # Derive map with next goal from goal category and oracle map with goals and/or distractors
                obj_cat_map = global_map[:,:,:,1]
                distrIndexOffset = 1 if self.agent_type == "oracle" else 2 # to match categories in oracle map
                goal_map = (obj_cat_map == (target_encoding + distrIndexOffset).unsqueeze(1)).type(torch.LongTensor)
                #global_map_embedding.append(self.object_embedding(goal_map.to(self.device).view(-1)).view(bs, 50, 50, -1))
                global_map_embedding.append(self.curr_goal_embedding(goal_map.to(self.device).view(-1)).view(bs, 50, 50, -1))
                if self.config.RL.MAPS.ALL_GOAL_IND:
                    global_map_embedding.append(self.object_embedding(obj_cat_map.type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            else:
                global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            
            if self.config.TASK_CONFIG.TASK.INCLUDE_DISTRACTORS and self.config.RL.MAPS.INCLUDE_DISTRACTOR_EMBED:
                global_map_embedding.append(self.distractor_embedding(global_map[:, :, :, 2].type(torch.LongTensor)
                                                                        .to(self.device).view(-1)).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x

        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states.permute(1, 0, 2), masks)
        return x, rnn_hidden_states.permute(1, 0, 2)  

class BaselineNetObjectRecog(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, device, 
        object_category_embedding_size, previous_action_embedding_size, use_previous_action
    ):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._n_input_goal = observation_space.spaces[
            self.goal_sensor_uuid
        ].shape[0]
        self._hidden_size = hidden_size
        self.device = device
        self.use_previous_action = use_previous_action

        self.visual_encoder = RGBCNNOracle(observation_space, 512)
        #self.visual_encoder = SimpleCNN(observation_space, 512)
        _n_input_map = 32
        self.map_encoder = MapCNN(50, 256, "obj-recog", _n_input_map)

        if self.use_previous_action:
            self.state_encoder = RNNStateEncoder(
                self._hidden_size + 256 + object_category_embedding_size + 
                previous_action_embedding_size, self._hidden_size,
            )
        else:
            self.state_encoder = RNNStateEncoder(
                (0 if self.is_blind else self._hidden_size) + object_category_embedding_size,
                self._hidden_size,   #Replace 2 by number of target categories later
            )
        self.goal_embedding = nn.Embedding(9, object_category_embedding_size)
        self.action_embedding = nn.Embedding(4, previous_action_embedding_size)

        self.occupancy_embedding = nn.Embedding(3, 16)
        self.object_embedding = nn.Embedding(9, 16)
        # self.exploration_embedding = nn.Embedding(3, 16)
        self.obj_linear = nn.Linear(512, 10) # Number of object categories + 1
        nn.init.orthogonal_(self.obj_linear.weight, gain=0.01)
        nn.init.constant_(self.obj_linear.bias, 0)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_encoding(self, observations):
        return observations[self.goal_sensor_uuid]

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        target_encoding = self.get_target_encoding(observations)
        x = [self.goal_embedding((target_encoding).type(torch.LongTensor).to(self.device)).squeeze(1)]
        bs = target_encoding.shape[0]
        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            x = [perception_embed] + x
            
            global_map_embedding = []
            global_map = observations['semMap'] # Oracle Map with occupancy and goals and/or distractors
            global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
            global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            # global_map_embedding.append(self.exploration_embedding(global_map[:, :, :, 2].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=3)
            map_embed = self.map_encoder(global_map_embedding)
            x = [map_embed] + x
            # x = [observations["heading"]] + x
        # global_map_embedding = []
        # global_map = torch.zeros((bs, 50, 50, 3)).to(self.device)
        # global_map_embedding.append(self.occupancy_embedding(global_map[:, :, :, 0].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50 , -1))
        # global_map_embedding.append(self.object_embedding(global_map[:, :, :, 1].type(torch.LongTensor).to(self.device).view(-1)).view(bs, 50, 50, -1))
        # global_map_embedding.append(global_map[:, :, :, 2].unsqueeze(3))
        # global_map_embedding = torch.cat(global_map_embedding, dim=3)
        if self.use_previous_action:
            x = torch.cat(x + [self.action_embedding(prev_actions).squeeze(1)], dim=1)
        else:
            x = torch.cat(x, dim=1)

        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states.permute(1, 0, 2), masks)
        obj_prob = self.obj_linear(perception_embed)

        return x, rnn_hidden_states.permute(1, 0, 2), obj_prob

class Policy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions, policy_config=None):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        if policy_config is None:
            self.action_distribution_type = "categorical"
        else:
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )

        if self.action_distribution_type == "categorical":
            self.action_distribution = CategoricalNet(
                self.net.output_size, self.dim_actions
            )
        elif self.action_distribution_type == "gaussian":
            self.action_distribution = GaussianNet(
                self.net.output_size,
                self.dim_actions,
                policy_config.ACTION_DIST,
            )
        else:
            ValueError(
                f"Action distribution {self.action_distribution_type}"
                "not supported."
            )

        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            if self.action_distribution_type == "categorical":
                action = distribution.mode()
            elif self.action_distribution_type == "gaussian":
                action = distribution.mean
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config, observation_space, action_space):
        pass
