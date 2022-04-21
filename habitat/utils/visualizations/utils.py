#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0,0,0]], dtype=np.uint8
                )

cv2 = try_cv2_import()


def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    if verbose:
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view


def tile_images(render_obs_images: List[np.ndarray]) -> np.ndarray:
    """Tiles multiple images of non-equal size to a single image. Images are
    tiled into columns making the returned image wider than tall.
    """
    # Get the images in descending order of vertical height.
    render_obs_images = sorted(
        render_obs_images, key=lambda x: x.shape[0], reverse=True
    )
    img_cols = [[render_obs_images[0]]]
    max_height = render_obs_images[0].shape[0]
    cur_y = 0.0
    # Arrange the images in columns with the largest image to the left.
    col = []
    for im in render_obs_images[1:]:
        if cur_y + im.shape[0] <= max_height:
            col.append(im)
            cur_y += im.shape[0]
        else:
            img_cols.append(col)
            col = [im]
            cur_y = im.shape[0]
    img_cols.append(col)
    col_widths = [max(col_ele.shape[1] for col_ele in col) for col in img_cols]
    # Get the total width of all the columns put together.
    total_width = sum(col_widths)

    # Tile the images, pasting the columns side by side.
    final_im = np.zeros(
        (max_height, total_width, 3), dtype=render_obs_images[0].dtype
    )
    cur_x = 0
    for i in range(len(img_cols)):
        next_x = cur_x + col_widths[i]
        total_col_im = np.concatenate(img_cols[i], axis=0)
        final_im[: total_col_im.shape[0], cur_x:next_x] = total_col_im
        cur_x = next_x
    return final_im

def draw_subsuccess(view: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    a subsuccess event has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view


def draw_found(view: np.ndarray, alpha: float = 1) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    that a found action has been called.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with found action effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view

def observations_to_image(observation: Dict, projected_features: np.ndarray=None, egocentric_projection: np.ndarray=None, global_map: np.ndarray=None, info: Dict=None, action: np.ndarray=None) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)
    
    if projected_features is not None and len(projected_features)>0:
        projected_features = cv2.resize(
            projected_features,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )
        projected_features /= np.max(projected_features)
        projected_features  = cv2.applyColorMap(np.uint8(255 * projected_features), cv2.COLORMAP_JET)
        egocentric_view.append(projected_features)

    if egocentric_projection is not None and len(egocentric_projection)>0:
        egocentric_projection = cv2.resize(
            egocentric_projection,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(egocentric_projection)
        
    if global_map is not None and len(global_map)>0:
        global_map = cv2.resize(
            global_map,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(global_map)
    
    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"] is not None and info["collisions"]["is_collision"] is not None:
        egocentric_view = draw_collision(egocentric_view)

    if action == 0:
        egocentric_view = draw_found(egocentric_view)

    frame = egocentric_view

    if "top_down_map" in info and info["top_down_map"] is not None and info["top_down_map"]["map"] is not None:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
        
        # Debug Oracle Maps
        if "global_occ_map" in observation:
            map = observation["global_occ_map"]
            map_size = map.shape
            currPix = observation["agent_position_grid"]
            map[currPix[0],currPix[1]] = 2
            tmp = np.transpose(np.where(map==100))
            for t in tmp:
                map[t[0],t[1]] = 2
            global_occ_map = recolor_map[map.astype(np.uint8)]
            global_occ_map = maps.draw_agent(
                image=global_occ_map,
                agent_center_coord=currPix,
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=2, #global_occ_map.shape[0] // 28,
            )

            global_occ_map = cv2.resize(
                global_occ_map,
                (top_down_width, top_down_height-10),
                interpolation=cv2.INTER_NEAREST,
            )
            bordersize = 5
            global_occ_map = cv2.copyMakeBorder(
                global_occ_map,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[200, 0, 0]
            )
            global_occ_map = cv2.putText(global_occ_map, f'Global_{map_size}', 
                                         (top_down_width-200, top_down_height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                                         0.7, (255, 0, 0), 2, cv2.LINE_AA)
            frame = np.concatenate((frame, global_occ_map), axis=1)
            
        if "local_occ_map" in observation:
            map = observation["local_occ_map"]
            map_size = map.shape
            if len(np.where(map==100)[0]) > 0:
                tmp = np.transpose(np.where(map==100))
                for t in tmp:
                    map[t[0],t[1]] = 2
                    currPix = t
                    map[currPix[0],currPix[1]] = 2
            else:
                currPix = (40,40)
                
            local_occ_map = recolor_map[map.astype(np.uint8)]
            
            local_occ_map = maps.draw_agent(
                image=local_occ_map,
                agent_center_coord=(local_occ_map.shape[0] // 2, local_occ_map.shape[1] // 2),
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=2,
            )

            #ocal_occ_map = np.rot90(local_occ_map, 1)

            local_occ_map = cv2.resize(
                local_occ_map,
                (top_down_width, top_down_height-10),
                interpolation=cv2.INTER_NEAREST,
            )
            
            bordersize = 5
            local_occ_map = cv2.copyMakeBorder(
                local_occ_map,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[200, 0, 0]
            )
            local_occ_map = cv2.putText(local_occ_map, f'Cropped_{map_size}', 
                                         (top_down_width-200, top_down_height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                                         0.7, (255, 0, 0), 2, cv2.LINE_AA)
            frame = np.concatenate((frame, local_occ_map), axis=1)
        
        """ if "local_rot_occ_map" in observation:
            map = observation["local_rot_occ_map"]
            map_size = map.shape
            if len(np.where(map==100)[0]) > 0:
                currPix = np.transpose(np.where(map==100))[0] #observation["agent_position_grid"]
                map[currPix[0],currPix[1]] = 2
            tmp = np.transpose(np.where(map==100))
            for t in tmp:
                map[t[0],t[1]] = 2
            occ_map = recolor_map[map.astype(np.uint8)]
            
            if len(np.where(map==100)[0]) > 0:
                occ_map = maps.draw_agent(
                    image=occ_map,
                    agent_center_coord=currPix,
                    agent_rotation=info["top_down_map"]["agent_angle"],
                    agent_radius_px=occ_map.shape[0] // 16,
                )
                
            old_h, old_w, _ = occ_map.shape
            top_down_height = observation_size
            top_down_width = int(float(top_down_height) / old_h * old_w)
            occ_map = cv2.resize(
                occ_map,
                (top_down_width, top_down_height),
                interpolation=cv2.INTER_NEAREST,
            )
            occ_map = cv2.putText(occ_map, f'Rotated_{map_size}', 
                                         (top_down_width-200, top_down_height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                                         0.7, (255, 0, 0), 2, cv2.LINE_AA)
            frame = np.concatenate((frame, occ_map), axis=1) """
            
        if "semMap" in observation:
            map = observation["semMap"][:,:,0]
            map_size = map.shape
            if len(np.where(map==100)[0]) > 0:
                currPix = np.transpose(np.where(map==100))[0] #observation["agent_position_grid"]
                map[currPix[0],currPix[1]] = 2
            tmp = np.transpose(np.where(map==100))
            for t in tmp:
                map[t[0],t[1]] = 2
            occ_map = recolor_map[map.astype(np.uint8)]
            
            #if len(np.where(map==100)[0]) > 0:
            occ_map = maps.draw_agent(
                image=occ_map,
                agent_center_coord=(occ_map.shape[0] // 2, occ_map.shape[1] // 2),
                agent_rotation=info["top_down_map"]["agent_angle"],
                agent_radius_px=2,
            )
            
            #occ_map = np.rot90(occ_map, 1)
            occ_map = cv2.resize(
                occ_map,
                (top_down_width, top_down_height-10),
                interpolation=cv2.INTER_NEAREST,
            )

            bordersize = 5
            occ_map = cv2.copyMakeBorder(
                occ_map,
                top=bordersize,
                bottom=bordersize,
                left=bordersize,
                right=bordersize,
                borderType=cv2.BORDER_CONSTANT,
                value=[200, 0, 0]
            )
            occ_map = cv2.putText(occ_map, f'Rotated_{map_size}', 
                                         (top_down_width-200, top_down_height-50), cv2.FONT_HERSHEY_SIMPLEX, 
                                         0.7, (255, 0, 0), 2, cv2.LINE_AA)
            frame = np.concatenate((frame, occ_map), axis=1)
        
    return frame

def append_text_to_image(image: np.ndarray, text: str):
    r"""Appends text underneath an image of size (height, width, channels).
    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.
    Args:
        image: the image to put text underneath
        text: a string to display
    Returns:
        A new image with text inserted underneath the input image
    """
    h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final
