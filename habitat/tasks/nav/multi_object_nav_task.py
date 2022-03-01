#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional

import math
import attr
import numpy as np
from gym import spaces
import random
import magnum as mn

import habitat_sim
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.embodied_task import Action, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.core.spaces import EmptySpace
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

try:
    from habitat.datasets.multi_object_nav.multi_object_nav_dataset import (
        MultiObjectNavDatasetV1,
    )
except ImportError:
    pass


@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoal(NavigationGoal):
    r"""Multi object goal provides information about a target object which is specified by object_category and position.
    Args:
        object_category: category that can be used to retrieve object
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        position: position of the object goal
    """

    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None


@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoalNavEpisode(NavigationEpisode):
    r"""Multi Object Goal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[List[str]] = None
    goals: List[Any] = []
    distractors: List[Any] = []


@registry.register_task(name="MultiObjectNav-v1")
class MultiObjectNavigationTask(NavigationTask):
    r"""A Multi Object Navigation Task class."""

    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)

    def reset(self, episode: MultiObjectGoalNavEpisode):
        # Remove existing objects from last episode
        for objid in self._sim.get_existing_object_ids():
            self._sim.remove_object(objid)

        # Insert current episode objects
        obj_templates_mgr = self._sim.get_object_template_manager()
        obj_type = self._config.OBJECTS_TYPE
        if obj_type == "CYL":
            obj_path = self._config.CYL_OBJECTS_PATH
        else:
            obj_path = self._config.REAL_OBJECTS_PATH
            
        for i in range(len(episode.goals)):
            current_goal = episode.goals[i].object_category
            object_index = obj_templates_mgr.load_configs(
                str(os.path.join(obj_path, current_goal))
            )[0]
            ind = self._sim.add_object(object_index)
            self._sim.set_translation(np.array(episode.goals[i].position), ind)
            
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            self._sim.set_rotation(y_rotation, ind)
            
            self._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)
            
        
        for i in range(len(episode.distractors)):
            current_distractor = episode.distractors[i].object_category
            dataset_index = obj_templates_mgr.load_configs(
                str(os.path.join(obj_path, current_distractor))
            )[0]
            
            ind = self._sim.add_object(dataset_index)
            self._sim.set_translation(np.array(episode.distractors[i].position), ind)
            
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            self._sim.set_rotation(y_rotation, ind)
            self._sim.set_object_motion_type(habitat_sim.physics.MotionType.STATIC, ind)

        # Reinitialize current goal index
        self.current_goal_index = 0

        # Initialize self.is_found_called
        self.is_found_called = False

        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        return observations

    def step(self, action: Dict[str, Any], episode):
        if "action_args" not in action or action["action_args"] is None:
            action["action_args"] = {}
        action_name = action["action"]

        self.is_found_called = bool(action_name == 0)

        if isinstance(action_name, (int, np.integer)):
            action_name = self.get_action_name(action_name)
        assert (
            action_name in self.actions
        ), f"Can't find '{action_name}' action in {self.actions.keys()}."

        task_action = self.actions[action_name]
        observations = task_action.step(**action["action_args"], task=self)
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations,
                episode=episode,
                action=action,
                task=self,
            )
        )

        self._is_episode_active = self._check_episode_is_active(
            episode=episode
        )

        return observations

    def _check_episode_is_active(
        self, episode: MultiObjectGoalNavEpisode
    ) -> bool:
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_current_goal = self._sim.geodesic_distance(
            current_position,
            episode.goals[self.current_goal_index].position,
        )
        is_episode_active = not (
            self.is_found_called
            and distance_to_current_goal > self._config.SUCCESS_DISTANCE
        )
        return is_episode_active


@registry.register_sensor
class MultiObjectGoalSensor(Sensor):
    r"""A sensor for Multi Object Goal specification as observations which is used in
    MultiObjectGoalNavigation. The goal is expected to be specified by object_id.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Multi Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "multiobjectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "MultiObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "multiobjectgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = max(self._dataset.category_to_task_category_id.values())

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ) -> Optional[int]:
        category_name = [i.object_category for i in episode.goals]
        goals = np.array(
            [
                self._dataset.category_to_task_category_id[i]
                for i in category_name
            ],
            dtype=np.int64,
        )
        return goals[task.current_goal_index : task.current_goal_index + 1]


@registry.register_measure
class DistanceToCurrentObjectGoal(Measure):
    """Calculates distance from the current object goal."""

    cls_uuid: str = "distance_to_current_object_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(
        self,
        episode,
        task,
        *args: Any,
        **kwargs: Any,
    ):
        self._metric = None
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_current_goal = self._sim.geodesic_distance(
            current_position, episode.goals[task.current_goal_index].position
        )

        self._metric = distance_to_current_goal


@registry.register_measure
class CurrentGoalSuccess(Measure):
    r"""Whether or not the agent succeeded in finding it's
    current goal. This measure depends on DistanceToCurrentObjectGoal measure.
    """

    cls_uuid: str = "current_goal_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrentObjectGoal.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self,
        *args: Any,
        episode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        distance_to_current_goal = task.measurements.measures[
            DistanceToCurrentObjectGoal.cls_uuid
        ].get_metric()

        if (
            task.is_found_called
            and distance_to_current_goal <= self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1
            task.current_goal_index += 1
            task.foundDistance = distance_to_current_goal
        else:
            self._metric = 0


@registry.register_measure
class Progress(Measure):
    r"""What fraction of goals are found. This measure depends on
    CurrentGoalSuccess measure.
    """

    cls_uuid: str = "progress"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [CurrentGoalSuccess.cls_uuid]
        )
        self._metric = 0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        ep_current_goal_success = task.measurements.measures[
            CurrentGoalSuccess.cls_uuid
        ].get_metric()

        if ep_current_goal_success:
            self._metric += 1 / len(episode.goals)


@registry.register_measure
class MultiONSuccess(Measure):
    r"""Whether or not the agent succeeded at its task
    This measure depends on CurrentGoalSuccess measure.
    """

    cls_uuid: str = "multiON_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [CurrentGoalSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        current_goal_success = task.measurements.measures[
            CurrentGoalSuccess.cls_uuid
        ].get_metric()

        if current_goal_success == 1 and task.current_goal_index == len(
            episode.goals
        ):
            self._metric = 1
        else:
            self._metric = 0


@registry.register_measure
class MultiONSPL(Measure):
    r"""SPL (Success weighted by Path Length) for sequential goals.

    ref: MultiON: Benchmarking Semantic Map Memory using Multi-Object
    Navigation - Wani et. al
    https://arxiv.org/abs/2012.03912
    The measure depends on MultiONSuccess measure.
    """

    cls_uuid: str = "multiON_spl"
    
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._agent_episode_distance: Optional[float] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        episode: MultiObjectGoalNavEpisode,
        task,
        *args: Any,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid, [MultiONSuccess.cls_uuid]
        )

        self._agent_episode_distance = 0.0
        self._previous_position = self._sim.get_agent_state().position
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self,
        *args: Any,
        episode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        ep_success = task.measurements.measures[
            MultiONSuccess.cls_uuid
        ].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )


@registry.register_measure
class MultiONPPL(Measure):
    r"""PPL (Progress weighted by Path Length) for sequential goals

    ref: MultiON: Benchmarking Semantic Map Memory using Multi-Object
    Navigation - Wani et. al
    https://arxiv.org/abs/2012.03912
    The measure depends on CurrentGoalSuccess measure.
    """
    cls_uuid: str = "multiON_ppl"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._agent_episode_distance: Optional[float] = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(
        self,
        episode: MultiObjectGoalNavEpisode,
        task,
        *args: Any,
        **kwargs: Any,
    ):
        task.measurements.check_measure_dependencies(
            self.uuid, [Progress.cls_uuid]
        )

        self._previous_position = self._sim.get_agent_state().position
        self._agent_episode_distance = 0.0
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._subsequent_goal_distances: List[float] = []
        subsequent_goal_distance = 0.0
        for goal_number in range(len(episode.goals)):
            if goal_number == 0:
                subsequent_goal_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
            else:
                subsequent_goal_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position,
                    episode.goals[goal_number].position,
                )
            self._subsequent_goal_distances.append(subsequent_goal_distance)

        self._metric = 0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)

    def update_metric(
        self,
        *args: Any,
        episode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        ep_progress = task.measurements.measures[
            Progress.cls_uuid
        ].get_metric()

        current_position = self._sim.get_agent_state().position
        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        if ep_progress > 0:
            self._metric = ep_progress * (
                self._subsequent_goal_distances[task.current_goal_index - 1]
                / max(
                    self._agent_episode_distance,
                    self._subsequent_goal_distances[
                        task.current_goal_index - 1
                    ],
                )
            )


@registry.register_measure
class DistanceToMultiGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_multi_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        """if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                # for goal in episode.goals     # Considering only one goal for now
                for view_point in episode.goals[episode.object_index][0].view_points
            ]"""
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        if self._config.DISTANCE_TO == "POINT":
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[task.current_goal_index].position
            )
            for goal_number in range(task.current_goal_index, len(episode.goals)-1):
                distance_to_target += self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )

        self._metric = distance_to_target

@registry.register_measure
class Ratio(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "ratio"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.DISTANCE_TO == "POINT":
            initial_geodesic_distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_geodesic_distance_to_target += self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )

            initial_euclidean_distance_to_target = self._euclidean_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_euclidean_distance_to_target += self._euclidean_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
        # else:
        #     logger.error(
        #         f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
        #     )
        self._metric = initial_geodesic_distance_to_target / initial_euclidean_distance_to_target

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        pass


@registry.register_measure
class EpisodeLength(Measure):
    r"""Calculates the episode length
    """
    cls_uuid: str = "episode_length"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        self._episode_length += 1
        self._metric = self._episode_length


@registry.register_measure
class RawMetrics(Measure):
    """All the raw metrics we might need
    """
    cls_uuid: str = "raw_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        self._start_subgoal_episode_distance = []
        self._start_subgoal_agent_distance = []
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)

        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [EpisodeLength.cls_uuid, MultiONSPL.cls_uuid, 
                        DistanceToMultiGoal.cls_uuid, DistanceToCurrentObjectGoal.cls_uuid,
                        MultiONSuccess.cls_uuid, Progress.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        task: MultiObjectNavigationTask,
        **kwargs: Any,
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_success = task.measurements.measures[MultiONSuccess.cls_uuid].get_metric()
        progress = task.measurements.measures[Progress.cls_uuid].get_metric()
        distance_to_curr_subgoal = task.measurements.measures[DistanceToCurrentObjectGoal.cls_uuid].get_metric()
        curr_goal_success = task.measurements.measures[CurrentGoalSuccess.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        if curr_goal_success:
            self._start_subgoal_agent_distance.append(self._agent_episode_distance)

        self._metric = {
            'success': ep_success,
            'progress': progress,
            'geodesic_distances': self._start_subgoal_episode_distance,
            'agent_path_length': self._agent_episode_distance,
            'subgoals_found': task.current_goal_index,
            'current_goal_success': curr_goal_success,
            'distance_to_curr_subgoal': distance_to_curr_subgoal,
            'agent_distances': self._start_subgoal_agent_distance,
            'distance_to_multi_goal': task.measurements.measures[DistanceToMultiGoal.cls_uuid].get_metric(),
            'spl': task.measurements.measures[MultiONSPL.cls_uuid].get_metric(),
            'ppl': task.measurements.measures[MultiONPPL.cls_uuid].get_metric(),
            'episode_lenth': task.measurements.measures[EpisodeLength.cls_uuid].get_metric()
        }


@registry.register_task_action
class FoundObjectAction(Action):
    name: str = "FOUND"

    def __init__(self, *args: Any, sim, dataset, **kwargs: Any) -> None:
        self._sim = sim

    def reset(
        self, task: MultiObjectNavigationTask, *args: Any, **kwargs: Any
    ) -> None:
        return

    def step(self, *args: Any, **kwargs: Any) -> Dict[str, Observations]:
        return self._sim.get_observations_at()

    @property
    def action_space(self) -> spaces.Dict:
        return EmptySpace()