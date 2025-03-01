#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from cmath import nan
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import pickle
import gym
import numba
import numpy as np
from gym import spaces
from scipy import ndimage
import math
import torch
import torchvision.transforms.functional as TF

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks import make_task
from habitat.utils import profiling_wrapper
import habitat_sim
import magnum as mn
from habitat.utils.visualizations import maps
from PIL import Image

# debug
COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -62.3241 - COORDINATE_EPSILON
COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON

class Env:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied tasks with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: Config
    _dataset: Optional[Dataset]
    number_of_episodes: Optional[int]
    _episodes: List[Episode]
    _current_episode_index: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        assert config.is_frozen(), (
            "Freeze the config before creating the "
            "environment, use config.freeze()."
        )
        self._config = config
        self._dataset = dataset
        self._current_episode_index = None
        if self._dataset is None and config.DATASET.TYPE:
            self._dataset = make_dataset(
                id_dataset=config.DATASET.TYPE, config=config.DATASET
            )
        self._episodes = (
            self._dataset.episodes
            if self._dataset
            else cast(List[Episode], [])
        )
        self._current_episode = None
        iter_option_dict = {
            k.lower(): v
            for k, v in config.ENVIRONMENT.ITERATOR_OPTIONS.items()
        }
        iter_option_dict["seed"] = config.SEED
        self._episode_iterator = self._dataset.get_episode_iterator(
            **iter_option_dict
        )

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._config.defrost()
            self._config.SIMULATOR.SCENE = self._dataset.episodes[0].scene_id
            self._config.freeze()

            self.number_of_episodes = len(self._dataset.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(
            id_sim=self._config.SIMULATOR.TYPE, config=self._config.SIMULATOR
        )
        self._task = make_task(
            self._config.TASK.TYPE,
            config=self._config.TASK,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = (
            self._config.ENVIRONMENT.MAX_EPISODE_SECONDS
        )
        self._max_episode_steps = self._config.ENVIRONMENT.MAX_EPISODE_STEPS
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False
        self._map_resolution = 300
        self.meters_per_pixel = self._config.TASK["MAP_RESOLUTION"] if "MAP_RESOLUTION" in self._config.TASK else 0.3
        self.cropped_map_size = self._config.TASK["CROP_MAP_SIZE"] if "CROP_MAP_SIZE" in self._config.TASK else 100
        self.egocentric_map_size = self._config.TASK["EGO_MAP_SIZE"] if "EGO_MAP_SIZE" in self._config.TASK else 51
        self.cache_map = self._config.TASK["CACHE_MAP"] if "CACHE_MAP" in self._config.TASK else True
        if config.TRAINER_NAME in ["oracle", "oracle-ego", "semantic"]:
            self.mapCache = {}
            # if self.old_map:
            #     with open(config.TASK.ORACLE_MAP_PATH, 'rb') as handle:
            #         self.mapCache = pickle.load(handle)
        if config.TRAINER_NAME == "oracle-ego":
            for x,y in self.mapCache.items():
                self.mapCache[x] += 1
        if config.TRAINER_NAME == "obj-recog":
            with open(config.TASK.ORACLE_MAP_PATH, 'rb') as handle:
                self.mapCache = pickle.load(handle)
            self.objGraph = np.empty((300,300,3))
            self.objGraph.fill(0)

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode

    @property
    def episode_iterator(self) -> Iterator:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator) -> None:
        self._episode_iterator = new_iter

    @property
    def episodes(self) -> List[Episode]:
        return self._episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert (
            len(episodes) > 0
        ), "Environment doesn't accept empty episodes list."
        self._episodes = episodes

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True
        elif (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        ):
            return True
        return False

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False
    
    def conv_grid(
        self,
        realworld_x,
        realworld_y,
        coordinate_min = COORDINATE_MIN, #-120.3241-1e-6,
        coordinate_max = COORDINATE_MAX, #120.0399+1e-6,
        grid_resolution = (300, 300)
    ):
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        grid_x = int((coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - coordinate_min) / grid_size[1])
        return grid_x, grid_y
    
    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"
        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        self._current_episode = next(self._episode_iterator)
        self.reconfigure(self._config)
        
        # Remove existing objects from last episode
        for objid in self._sim.get_existing_object_ids():
            self._sim.remove_object(objid)

        # Insert object here
        obj_type = self._config["TASK"]["OBJECTS_TYPE"]
        if obj_type == "CYL":
            self.object_to_datset_mapping = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2, 'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7}
        else:
            self.object_to_datset_mapping = {'guitar':0, 'electric_piano':1, 'basket_ball':2,'toy_train':3, 'teddy_bear':4, 'rocking_horse':5, 'backpack': 6, 'trolley_bag':7}
            
            
        for i in range(len(self.current_episode.goals)):
            current_goal = self.current_episode.goals[i].object_category
            dataset_index = self.object_to_datset_mapping[current_goal]
            ind = self._sim.add_object(dataset_index)
            self._sim.set_object_semantic_id(dataset_index, ind)
            self._sim.set_translation(np.array(self.current_episode.goals[i].position), ind)
            
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            self._sim.set_rotation(y_rotation, ind)
            self._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)

        if self._config["TASK"]["INCLUDE_DISTRACTORS"]:
            for i in range(len(self.current_episode.distractors)):
                current_distractor = self.current_episode.distractors[i].object_category
                dataset_index = self.object_to_datset_mapping[current_distractor]
                ind = self._sim.add_object(dataset_index)
                self._sim.set_object_semantic_id(dataset_index, ind)
                self._sim.set_translation(np.array(self.current_episode.distractors[i].position), ind)
                
                # random rotation only on the Y axis
                y_rotation = mn.Quaternion.rotation(
                    mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
                )
                self._sim.set_rotation(y_rotation, ind)
                self._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)

        observations = self.task.reset(episode=self.current_episode)
        if self._config.TRAINER_NAME in ["oracle", "oracle-ego", "obj-recog", "semantic"]:
            #self.currMap = np.copy(self.mapCache[f"../multiON/{self.current_episode.scene_id}"])
            agent_vertical_pos = str(round(observations["agent_position"][1],2))
            if self.cache_map and (self.current_episode.scene_id in self.mapCache and 
                    agent_vertical_pos in self.mapCache[self.current_episode.scene_id]):
                top_down_map = (self.mapCache[self.current_episode.scene_id][agent_vertical_pos]).copy()
                tmp_map = np.zeros((top_down_map.shape[0],top_down_map.shape[1],3))
                tmp_map[:top_down_map.shape[0], :top_down_map.shape[1], 0] = top_down_map
                self.currMap = tmp_map
            else:
                # topdown map obtained from habitat has 0 if occupied, 1 if unoccupied
                top_down_map = maps.get_topdown_map_from_sim(
                    self._sim,
                    draw_border=False,
                    meters_per_pixel=self.meters_per_pixel,
                    with_sampling=True,
                    num_samples=100
                )
                range_x = np.where(np.any(top_down_map, axis=1))[0]
                range_y = np.where(np.any(top_down_map, axis=0))[0]
                padding = int(np.ceil(top_down_map.shape[0] / 400))
                range_x = (
                    max(range_x[0] - padding, 0),
                    min(range_x[-1] + padding + 1, top_down_map.shape[0]),
                )
                range_y = (
                    max(range_y[0] - padding, 0),
                    min(range_y[-1] + padding + 1, top_down_map.shape[1]),
                )
                
                # update topdown map to have 1 if occupied, 2 if unoccupied
                top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]] += 1
                
                if self._config.TRAINER_NAME in ["oracle-ego", "semantic"]:
                    top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]] += 1
                
                if self.current_episode.scene_id not in self.mapCache:
                    self.mapCache[self.current_episode.scene_id] = {}
                self.mapCache[self.current_episode.scene_id][agent_vertical_pos] = top_down_map
                
                tmp_map = np.zeros((top_down_map.shape[0],top_down_map.shape[1],3))
                tmp_map[:top_down_map.shape[0], :top_down_map.shape[1], 0] = top_down_map.copy()
                self.currMap = tmp_map
                
            #self.task.occMap = self.currMap[:,:,0]
            self.task.sceneMap = self.currMap[:,:,0]
        
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            channel_num = 1
            # Adding goal information on the map
            for i in range(len(self.current_episode.goals)):
                loc0 = self.current_episode.goals[i].position[0]
                loc2 = self.current_episode.goals[i].position[2]
                #grid_loc = self.conv_grid(loc0, loc2)
                grid_loc = maps.to_grid(
                    loc2,
                    loc0,
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
                objIndexOffset = 1 if self._config.TRAINER_NAME == "oracle" else 2
                self.currMap[grid_loc[0]-1:grid_loc[0]+2, grid_loc[1]-1:grid_loc[1]+2, channel_num] = self.object_to_datset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset
                
            if self._config["TASK"]["INCLUDE_DISTRACTORS"]:
                if self._config["TASK"]["ORACLE_MAP_INCLUDE_DISTRACTORS_W_GOAL"]:
                    channel_num = 1
                else:
                    channel_num = 2
                
                # Adding distractor information on the map
                distrIndexOffset = 1 if self._config.TRAINER_NAME == "oracle" else 2
                num_distr = self._config["TASK"]["NUM_DISTRACTORS"] if self._config["TASK"]["NUM_DISTRACTORS"] > 0 else len(self.current_episode.distractors)
                for i in range(num_distr):
                    loc0 = self.current_episode.distractors[i].position[0]
                    loc2 = self.current_episode.distractors[i].position[2]
                    #grid_loc = self.conv_grid(loc0, loc2)
                    grid_loc = maps.to_grid(
                        loc2,
                        loc0,
                        self.currMap.shape[0:2],
                        sim=self._sim,
                    )
                    self.currMap[grid_loc[0]-1:grid_loc[0]+2, grid_loc[1]-1:grid_loc[1]+2, channel_num] = self.object_to_datset_mapping[self.current_episode.distractors[i].object_category] + distrIndexOffset

            #currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking
            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
            
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = np.repeat(
                    self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis], 3, axis = 2
                )
                patch_tmp = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch_tmp = self.currMap
            
            patch = patch_tmp[max(currPix[0]-40,0):currPix[0]+40, max(currPix[1]-40,0):currPix[1]+40,:]
            if patch.shape[0] < 80 or patch.shape[1] < 80:
                if currPix[0] < 40:
                    curr_x = currPix[0]
                else:
                    curr_x = 40
                if currPix[1] < 40:
                    curr_y = currPix[1]
                else:
                    curr_y = 40
                    
                map_mid = (80//2)
                tmp = np.zeros((80, 80,3))
                tmp[map_mid-curr_x:map_mid-curr_x+patch.shape[0],
                        map_mid-curr_y:map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
                
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            #patch = TF.rotate(torch.tensor(patch).permute(2,0,1), -(observations["heading"][0] * 180/np.pi) + 90).permute(1,2,0).numpy()

            sem_map = patch[40-25:40+25, 40-25:40+25, :]
            observations["semMap"] = sem_map
        elif self._config.TRAINER_NAME in ["obj-recog"]:
            self.objGraph.fill(0)
            channel_num = 1
            # Adding goal information on the map
            for i in range(len(self.current_episode.goals)):
                loc0 = self.current_episode.goals[i].position[0]
                loc2 = self.current_episode.goals[i].position[2]
                #grid_loc = self.conv_grid(loc0, loc2)
                grid_loc = maps.to_grid(
                    loc2,
                    loc0,
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
                objIndexOffset = 1
                self.currMap[grid_loc[0]-1:grid_loc[0]+2, grid_loc[1]-1:grid_loc[1]+2, channel_num] = self.object_to_datset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset
            
                # Marking category of the goals
                self.objGraph[grid_loc[0]-3:grid_loc[0]+4, grid_loc[1]-3:grid_loc[1]+4, 0] = self.object_to_datset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset
                self.objGraph[grid_loc[0]-3:grid_loc[0]+4, grid_loc[1]-3:grid_loc[1]+4, 1] = loc0
                self.objGraph[grid_loc[0]-3:grid_loc[0]+4, grid_loc[1]-3:grid_loc[1]+4, 2] = loc2
                
            if self._config["TASK"]["INCLUDE_DISTRACTORS"]:
                if self._config["TASK"]["ORACLE_MAP_INCLUDE_DISTRACTORS_W_GOAL"]:
                    channel_num = 1
                else:
                    channel_num = 2
                
                # Adding distractor information on the map
                distrIndexOffset = 1
                num_distr = self._config["TASK"]["NUM_DISTRACTORS"] if self._config["TASK"]["NUM_DISTRACTORS"] > 0 else len(self.current_episode.distractors)
                for i in range(num_distr):
                    loc0 = self.current_episode.distractors[i].position[0]
                    loc2 = self.current_episode.distractors[i].position[2]
                    #grid_loc = self.conv_grid(loc0, loc2)
                    grid_loc = maps.to_grid(
                        loc2,
                        loc0,
                        self.currMap.shape[0:2],
                        sim=self._sim,
                    )
                    self.currMap[grid_loc[0]-1:grid_loc[0]+2, grid_loc[1]-1:grid_loc[1]+2, channel_num] = self.object_to_datset_mapping[self.current_episode.distractors[i].object_category] + distrIndexOffset

            #currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking
            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
            if self.currMap[currPix[0], currPix[1], 2] == 0:
                self.currMap[currPix[0], currPix[1], 2] = 1
            else:
                self.currMap[currPix[0], currPix[1], 2] = 2
                
            patch = self.currMap
            patch = patch_tmp[max(currPix[0]-40,0):currPix[0]+40, max(currPix[1]-40,0):currPix[1]+40,:]
            if patch.shape[0] < 80 or patch.shape[1] < 80:
                if currPix[0] < 40:
                    curr_x = currPix[0]
                else:
                    curr_x = 40
                if currPix[1] < 40:
                    curr_y = currPix[1]
                else:
                    curr_y = 40
                    
                map_mid = (80//2)
                tmp = np.zeros((80, 80,3))
                tmp[map_mid-curr_x:map_mid-curr_x+patch.shape[0],
                        map_mid-curr_y:map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
                
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            #patch = TF.rotate(torch.tensor(patch).permute(2,0,1), -(observations["heading"][0] * 180/np.pi) + 90).permute(1,2,0).numpy()
            
            sem_map = patch[40-25:40+25, 40-25:40+25, :]
            observations["semMap"] = sem_map
            
            # code for object category
            if self.objGraph[currPix[0], currPix[1], 0] != 0:
                vector = np.array([self.objGraph[currPix[0], currPix[1], 1], self.objGraph[currPix[0], currPix[1], 2]]) - \
                    np.array([observations["agent_position"][0], observations["agent_position"][2]])
                goalToAgentHeading = np.arctan2(-vector[0], -vector[1]) * 180 / np.pi
                includedAng = np.absolute((observations["heading"][0] * 180/np.pi) - goalToAgentHeading)
                if includedAng > 180.0:
                    includedAng = 360.0 -180.0
                assert includedAng >= 0
                assert includedAng <= 180.0
                if includedAng < 40.0: 
                    observations["objectCat"] = self.objGraph[currPix[0], currPix[1], 0]
                else:
                    observations["objectCat"] = 0
            else:
                observations["objectCat"] = 0
        elif self._config.TRAINER_NAME in ["semantic"]:
            channel_num = 1
            objIndexOffset = 1  # 0 -> no object

            # Adding goal information on the map
            for i in range(len(self.current_episode.goals)):
                goal_id = self.object_to_datset_mapping[self.current_episode.goals[i].object_category] + objIndexOffset
                #if goal_id not in observations["semantic"]:
                #    continue
                
                loc0 = self.current_episode.goals[i].position[0]
                loc2 = self.current_episode.goals[i].position[2]
                #grid_loc = self.conv_grid(loc0, loc2)
                grid_loc = maps.to_grid(
                    loc2,
                    loc0,
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
                self.currMap[
                    grid_loc[0]-1:grid_loc[0]+2, 
                    grid_loc[1]-1:grid_loc[1]+2, 
                    channel_num] = goal_id
                
            if self._config["TASK"]["INCLUDE_DISTRACTORS"]:
                # Adding distractor information on the map
                num_distr = self._config["TASK"]["NUM_DISTRACTORS"] if self._config["TASK"]["NUM_DISTRACTORS"] > 0 else len(self.current_episode.distractors)
                for i in range(num_distr):
                    distractor_id = self.object_to_datset_mapping[self.current_episode.distractors[i].object_category] + objIndexOffset
                 #   if distractor_id not in observations["semantic"]:
                  #      continue
                    
                    loc0 = self.current_episode.distractors[i].position[0]
                    loc2 = self.current_episode.distractors[i].position[2]
                    #grid_loc = self.conv_grid(loc0, loc2)
                    grid_loc = maps.to_grid(
                        loc2,
                        loc0,
                        self.currMap.shape[0:2],
                        sim=self._sim,
                    )
                    self.currMap[grid_loc[0]-1:grid_loc[0]+2, 
                                 grid_loc[1]-1:grid_loc[1]+2, 
                                 channel_num] = distractor_id

            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )

            patch_tmp = self.currMap
            
            cropped_map_mid = (self.cropped_map_size//2)
            # Crop around agent: agent is at the center
            patch = patch_tmp[max(currPix[0]-cropped_map_mid,0):currPix[0]+cropped_map_mid, 
                              max(currPix[1]-cropped_map_mid,0):currPix[1]+cropped_map_mid,:]
            if patch.shape[0] < self.cropped_map_size or patch.shape[1] < self.cropped_map_size:
                if currPix[0] < cropped_map_mid:
                    curr_x = currPix[0]
                else:
                    curr_x = cropped_map_mid
                if currPix[1] < cropped_map_mid:
                    curr_y = currPix[1]
                else:
                    curr_y = cropped_map_mid

                tmp = np.zeros((self.cropped_map_size, self.cropped_map_size, 3))
                tmp[cropped_map_mid-curr_x:cropped_map_mid-curr_x+patch.shape[0],
                        cropped_map_mid-curr_y:cropped_map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
                
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            
            # agent is on bottom center
            ego_map_mid = (self.egocentric_map_size//2)
            sem_map = patch[cropped_map_mid-ego_map_mid:cropped_map_mid+ego_map_mid+1, 
                            cropped_map_mid-1:cropped_map_mid+self.egocentric_map_size, 0]
            
            # occ_map = patch[cropped_map_mid-ego_map_mid:cropped_map_mid+ego_map_mid+1, 
            #                 cropped_map_mid-1:cropped_map_mid+self.egocentric_map_size, 0]
            
            # Image.fromarray(
            #     maps.colorize_topdown_map(
            #         (sem_map-1+maps.MULTION_TOP_DOWN_MAP_START).astype(np.uint8)
            #         )
            #     ).save(
            #     f"test_maps/{self.current_episode.episode_id}_new_0.png")
            self.count = 0
            observations["semMap"] = sem_map
            #observations["nextGoalMap"] = ((sem_map-3) == observations['multiobjectgoal'])
            #observations["occMap"] = occ_map
            
        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active
        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(
        self, action: Union[int, str, Dict[str, Any]], **kwargs
    ) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"

        """ if self._config.ORACLE_FOUND and self._episode_oracle_found():
            action = 0  # Oracle Found """
        
        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            self.task.is_found_called = bool(action == 0)
            action = {"action": action}
        else:
            if action["action"] == 0:
                print('action=0.')
            self.task.is_found_called = bool(action["action"] == 0)
            
        observations = self.task.step(
            action=action, episode=self.current_episode
        )

        self._task.measurements.update_measures(
            episode=self.current_episode,
            action=action,
            task=self.task,
            observations=observations,
        )

        if self._config.TRAINER_NAME in ["oracle", "oracle-ego"]:
            #currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking
            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
            
            if self._config.TRAINER_NAME == "oracle-ego":
                self.expose = np.repeat(
                    self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis], 3, axis = 2
                )
                patch_tmp = self.currMap * self.expose
            elif self._config.TRAINER_NAME == "oracle":
                patch_tmp = self.currMap

            patch = patch_tmp[max(currPix[0]-40,0):currPix[0]+40, max(currPix[1]-40,0):currPix[1]+40,:]
            if patch.shape[0] < 80 or patch.shape[1] < 80:
                if currPix[0] < 40:
                    curr_x = currPix[0]
                else:
                    curr_x = 40
                if currPix[1] < 40:
                    curr_y = currPix[1]
                else:
                    curr_y = 40
                    
                map_mid = (80//2)
                tmp = np.zeros((80, 80,3))
                tmp[map_mid-curr_x:map_mid-curr_x+patch.shape[0],
                        map_mid-curr_y:map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
            
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            #patch = TF.rotate(torch.tensor(patch).permute(2,0,1), -(observations["heading"][0] * 180/np.pi) + 90).permute(1,2,0).numpy()
            
            sem_map = patch[40-25:40+25, 40-25:40+25, :]
            observations["semMap"] = sem_map
        
        elif self._config.TRAINER_NAME in ["obj-recog"]:
            #currPix = self.conv_grid(observations["agent_position"][0], observations["agent_position"][2])  ## Explored area marking
            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
            if self.currMap[currPix[0], currPix[1], 2] == 0:
                self.currMap[currPix[0], currPix[1], 2] = 1
            else:
                self.currMap[currPix[0], currPix[1], 2] = 2
                
            patch = self.currMap
            patch = patch_tmp[max(currPix[0]-40,0):currPix[0]+40, max(currPix[1]-40,0):currPix[1]+40,:]
            if patch.shape[0] < 80 or patch.shape[1] < 80:
                if currPix[0] < 40:
                    curr_x = currPix[0]
                else:
                    curr_x = 40
                if currPix[1] < 40:
                    curr_y = currPix[1]
                else:
                    curr_y = 40
                    
                map_mid = (80//2)
                tmp = np.zeros((80, 80,3))
                tmp[map_mid-curr_x:map_mid-curr_x+patch.shape[0],
                        map_mid-curr_y:map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
                
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            #patch = TF.rotate(torch.tensor(patch).permute(2,0,1), -(observations["heading"][0] * 180/np.pi) + 90).permute(1,2,0).numpy()
            
            sem_map = patch[40-25:40+25, 40-25:40+25, :]
            observations["semMap"] = sem_map
            
            # code for objectCat
            if self.objGraph[currPix[0], currPix[1], 0] != 0:
                vector = np.array([self.objGraph[currPix[0], currPix[1], 1], self.objGraph[currPix[0], currPix[1], 2]]) - \
                    np.array([observations["agent_position"][0], observations["agent_position"][2]])
                goalToAgentHeading = np.arctan2(-vector[0], -vector[1]) * 180 / np.pi
                includedAng = np.absolute((observations["heading"][0] * 180/np.pi) - goalToAgentHeading)
                if includedAng > 180.0:
                    includedAng = 360.0 -180.0
                assert includedAng >= 0
                assert includedAng <= 180.0
                if includedAng < 40.0: 
                    observations["objectCat"] = self.objGraph[currPix[0], currPix[1], 0]
                else:
                    observations["objectCat"] = 0
            else:
                observations["objectCat"] = 0
        elif self._config.TRAINER_NAME in ["semantic"]:
            currPix = maps.to_grid(
                    observations["agent_position"][2],
                    observations["agent_position"][0],
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )

            patch_tmp = self.currMap
            
            cropped_map_mid = (self.cropped_map_size//2)
            # Crop around agent: agent is at the center
            patch = patch_tmp[max(currPix[0]-cropped_map_mid,0):currPix[0]+cropped_map_mid, 
                              max(currPix[1]-cropped_map_mid,0):currPix[1]+cropped_map_mid,:]
            if patch.shape[0] < self.cropped_map_size or patch.shape[1] < self.cropped_map_size:
                if currPix[0] < cropped_map_mid:
                    curr_x = currPix[0]
                else:
                    curr_x = cropped_map_mid
                if currPix[1] < cropped_map_mid:
                    curr_y = currPix[1]
                else:
                    curr_y = cropped_map_mid

                tmp = np.zeros((self.cropped_map_size, self.cropped_map_size, 3))
                tmp[cropped_map_mid-curr_x:cropped_map_mid-curr_x+patch.shape[0],
                        cropped_map_mid-curr_y:cropped_map_mid-curr_y+patch.shape[1], :] = patch
                patch = tmp
                
            patch = ndimage.interpolation.rotate(patch, -(observations["heading"][0] * 180/np.pi) + 90, axes=(0,1), order=0, reshape=False)
            
            # agent is on bottom center
            ego_map_mid = (self.egocentric_map_size//2)
            sem_map = patch[cropped_map_mid-ego_map_mid:cropped_map_mid+ego_map_mid+1, 
                            cropped_map_mid-1:cropped_map_mid+self.egocentric_map_size, 0]
            
            #occ_map = patch[cropped_map_mid-ego_map_mid:cropped_map_mid+ego_map_mid+1, 
            #                cropped_map_mid-1:cropped_map_mid+self.egocentric_map_size, 0]
            
            # Image.fromarray(
            #     maps.colorize_topdown_map(
            #         (sem_map-1+maps.MULTION_TOP_DOWN_MAP_START).astype(np.uint8)
            #         )
            #     ).save(
            #     f"test_maps/{self.current_episode.episode_id}_new_{self.count}.png")
            # self.count += 1
            observations["semMap"] = sem_map
            #observations["nextGoalMap"] = ((sem_map-3) == observations['multiobjectgoal'])
            #observations["occMap"] = occ_map
            
        ##Terminates episode if wrong found is called
        if self.task.is_found_called == True and \
            self.task.measurements.measures[
            "sub_success" #"current_goal_success"
        ].get_metric() == 0:
            self.task._is_episode_active = False
        
        ##Terminates episode if all goals are found
        if self.task.is_found_called == True and \
            self.task.current_goal_index == len(self.current_episode.goals):
            self.task._is_episode_active = False
            
        self._update_step_stats()

        return observations

    def _episode_oracle_found(self):
        return self._env.get_metrics()[self._config.RL.ORACLE_SUBSUCCESS_MEASURE]
    
    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: Config) -> None:
        self._config = config

        self._config.defrost()
        self._config.SIMULATOR = self._task.overwrite_sim_config(
            self._config.SIMULATOR, self.current_episode
        )
        self._config.freeze()

        self._sim.reconfigure(self._config.SIMULATOR)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """

        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    @property
    def current_episode(self) -> Episode:
        return self._env.current_episode

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations, **kwargs)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
