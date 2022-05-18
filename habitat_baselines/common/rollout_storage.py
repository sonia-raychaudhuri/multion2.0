#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch

from habitat_baselines.common.tensor_dict import TensorDict


class RolloutStorageNonOracle:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        global_map_size,
        global_map_depth,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.global_map = torch.zeros(
            num_steps + 1,
            num_envs,
            51,
            51,
            global_map_depth,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.global_map = self.global_map.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        global_map,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.global_map[self.step + 1].copy_(
            global_map
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.global_map[0].copy_(
            self.global_map[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)
            
            recurrent_hidden_states_batch = []
            global_map_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )
                global_map_batch.append(
                    self.global_map[1: self.step+1, ind]
                )
                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )
            global_map_batch = torch.stack(
                global_map_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )
            global_map_batch = self._flatten_helper(T, N, global_map_batch)
            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                global_map_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorageOracle:
    r"""Class for storing rollout information for RL trainers.

    """

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.observations["semMap"] = torch.zeros(
                num_steps+1, 
                num_envs,
                50, 50, 3
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1,
            num_recurrent_layers,
            num_envs,
            recurrent_hidden_state_size,
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations,
        recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.recurrent_hidden_states[0].copy_(
            self.recurrent_hidden_states[self.step]
        )
        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[self.step] = next_value
            gae = 0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step] = next_value
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            prev_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0, :, ind]
                )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[: self.step, ind]
                )

                adv_targ.append(advantages[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (num_recurrent_layers, N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            )

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            value_preds_batch = self._flatten_helper(T, N, value_preds_batch)
            return_batch = self._flatten_helper(T, N, return_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = self._flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = self._flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])
    
class RolloutStorageSemantic:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
        global_map_size:int = 275,
        global_map_depth:int = 32,
        local_map_size:int = 51,
        num_classes:int = 10
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["observations"]["semMap"] = torch.zeros(
                numsteps+1, 
                num_envs,
                local_map_size,
                local_map_size
            )

        # Stores cropped and rotated map
        self.buffers["global_map"] = torch.zeros(
            numsteps + 1,
            num_envs,
            local_map_size,
            local_map_size,
            global_map_depth,
        )
        
        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        global_map=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            global_map=global_map,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]
            batch["global_map"] = batch[
                "global_map"
            ][:self.current_rollout_step_idx]

            yield batch.map(lambda v: v.flatten(0, 1))
    

class RolloutStorageObjRecog:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["observations"]["semMap"] = torch.zeros(
                numsteps+1, 
                num_envs,
                50, 50, 3
            )

        self.buffers["observations"]["objectCat"] = torch.Tensor(
                numsteps+1, 
                num_envs
            ).fill_(0)
        
        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))
    
    
class RolloutStorage:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))

class RolloutStorageOracleMap:
    r"""Class for storing rollout information for RL trainers."""

    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
    ):
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["observations"]["semMap"] = torch.zeros(
                numsteps+1, 
                num_envs,
                50, 50, 3
            )
        
        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(
            numsteps + 1, num_envs, 1
        )

        if action_shape is None:
            if action_space.__class__.__name__ == "ActionSpace":
                action_shape = (1,)
            else:
                action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if (
            discrete_actions
            and action_space.__class__.__name__ == "ActionSpace"
        ):
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(
            numsteps + 1, num_envs, 1, dtype=torch.bool
        )

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    @property
    def current_rollout_step_idx(self) -> int:
        assert all(
            s == self.current_rollout_step_idxs[0]
            for s in self.current_rollout_step_idxs
        )
        return self.current_rollout_step_idxs[0]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
    ):
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
        )

        next_step = {k: v for k, v in next_step.items() if v is not None}
        current_step = {k: v for k, v in current_step.items() if v is not None}

        env_slice = slice(
            int(buffer_index * self._num_envs / self._nbuffers),
            int((buffer_index + 1) * self._num_envs / self._nbuffers),
        )

        if len(next_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index] + 1, env_slice),
                next_step,
                strict=False,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (self.current_rollout_step_idxs[buffer_index], env_slice),
                current_step,
                strict=False,
            )

    def advance_rollout(self, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1

    def after_update(self):
        self.buffers[0] = self.buffers[self.current_rollout_step_idx]

        self.current_rollout_step_idxs = [
            0 for _ in self.current_rollout_step_idxs
        ]

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.buffers["value_preds"][
                self.current_rollout_step_idx
            ] = next_value
            gae = 0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    self.buffers["rewards"][step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = (
                    delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                )
                self.buffers["returns"][step] = (
                    gae + self.buffers["value_preds"][step]
                )
        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    def recurrent_generator(self, advantages, num_mini_batch) -> TensorDict:
        num_environments = advantages.size(1)
        assert num_environments >= num_mini_batch, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(
                num_environments, num_mini_batch
            )
        )
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_mini_batch):
            batch = self.buffers[0 : self.current_rollout_step_idx, inds]
            batch["advantages"] = advantages[
                0 : self.current_rollout_step_idx, inds
            ]
            batch["recurrent_hidden_states"] = batch[
                "recurrent_hidden_states"
            ][0:1]

            yield batch.map(lambda v: v.flatten(0, 1))
    