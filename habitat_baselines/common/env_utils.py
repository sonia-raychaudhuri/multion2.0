#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import random
from typing import Type, Union

import habitat
from habitat import Config, Env, RLEnv, VectorEnv, ThreadedVectorEnv, make_dataset, logger


def make_env_fn(
    config: Config, env_class: Type[Union[Env, RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.
        rank: rank of env to be created (for seeding).

    Returns:
        env object created according to specification.
    """
    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env

def construct_envs(
    config: Config, env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.

    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
        env_class: class type of the envs to be created.
        :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor

    Returns:
        VectorEnv object created according to specification.
    """
    
    num_processes = config.NUM_PROCESSES
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE)
    scenes = config.TASK_CONFIG.DATASET.CONTENT_SCENES
    if "*" in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.TASK_CONFIG.DATASET)

    if num_processes > 1:
        if len(scenes) == 0:
            raise RuntimeError(
                "No scenes to load, multiple process logic relies on being able to split scenes uniquely between processes"
            )

        # if len(scenes) < num_processes:   # Allowing more workers per scene
        #     raise RuntimeError(
        #         "reduce the number of processes as there "
        #         "aren't enough number of scenes"
        #     )

        random.shuffle(scenes)

    scene_splits = [[] for _ in range(num_processes)]

    if len(scenes) >= num_processes:
        for idx, scene in enumerate(scenes):
            scene_splits[idx % len(scene_splits)].append(scene)
    else:
        sc=0
        for i in range(num_processes):
            scene_splits[i].append(scenes[sc%len(scenes)])
            sc+=1



    # assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        proc_config = config.clone()
        proc_config.defrost()

        task_config = proc_config.TASK_CONFIG
        task_config.SEED = task_config.SEED + i
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            config.SIMULATOR_GPU_ID
        )
                             
        task_config.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

        proc_config.freeze()
        configs.append(proc_config)

    logger.info(f"In construct_envs with {num_processes} processes. workers_ignore_signals={workers_ignore_signals}")
    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(tuple(zip(configs, env_classes))),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
