import sys
sys.path.insert(0, "")
import os
import json
import jsonlines
from collections import defaultdict
from tqdm import tqdm, trange

import numpy as np
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat import Config, logger, Agent
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.utils import generate_video
from habitat_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat.core.env import Env

EPSILON = 1e-6

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    # turn off RGBD rendering as neither RandomAgent nor HandcraftedAgent use it.
    #config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = []
    #config.TASK_CONFIG.TASK.SENSORS = []
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    """ if len(config.VIDEO_OPTION) > 0:
        config.defrost()
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        os.makedirs(config.VIDEO_DIR, exist_ok=True) """

    config.freeze()

    env = Env(config=config.TASK_CONFIG)

    assert config.EVAL.NONLEARNING.AGENT in [
        "OracleAgent",
    ], "EVAL.NONLEARNING.AGENT must be OracleAgent."

    agent = OracleAgent(config.TASK_CONFIG, env)

    stats = defaultdict(float)
    num_episodes = min(config.EVAL.EPISODE_COUNT, len(env.episodes))

    episode_stats = {}
    for _ in trange(num_episodes):
        obs = env.reset()
        agent.reset()

        rgb_frames = []
        while not env.episode_over:
            if len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(obs, info=env.get_metrics())
                rgb_frames.append(frame)

            action = agent.act(obs)
            obs = env.step(action)

        metrics_info = env.get_metrics()
        if len(config.VIDEO_OPTION) > 0:
            generate_video(
                video_option=config.VIDEO_OPTION,
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=env.current_episode.episode_id,
                checkpoint_idx=0,
                metrics={
                    "episode": float(env.current_episode.episode_id)
                },
                tb_writer=None,
            )

        if "top_down_map" in metrics_info:
            del metrics_info["top_down_map"]
        if "collisions" in metrics_info:
            del metrics_info["collisions"]
        
        episode_stats[env.current_episode.episode_id] = metrics_info
        
        for m, v in metrics_info.items():
            stats[m] += v

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(config.RESULTS_DIR, f"episode_stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json"), "w") as f:
        json.dump(episode_stats, f, indent=4)

    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(os.path.join(config.RESULTS_DIR, f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json"), "w") as f:
        json.dump(stats, f, indent=4)

class RandomAgent(Agent):
    r"""Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        if probs is not None:
            self.probs = probs
        else:
            self.probs = [0.02, 0.68, 0.15, 0.15]

    def reset(self):
        pass

    def act(self, observations):
        return {"action": np.random.choice(self.actions, p=self.probs)}


class HandcraftedAgent(Agent):
    r"""Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}

class OracleAgent(habitat.Agent):
    def __init__(self, task_config: habitat.Config, env):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.env = env

    def reset(self):
        self.follower = ShortestPathFollower(
            self.env.sim, goal_radius=0.25, return_one_hot=False
        )

    def act(self, observations):
        current_goal = self.env.task.current_goal_index
        self.env.sim.get_agent_state().position
        best_action = self.follower.get_next_action(self.env.current_episode.goals[current_goal].position)
                
        return best_action
