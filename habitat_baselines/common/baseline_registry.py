#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""BaselineRegistry is extended from habitat.Registry to provide
registration for trainer and environments, while keeping Registry
in habitat core intact.

Import the baseline registry object using

``from habitat_baselines.common.baseline_registry import baseline_registry``

Various decorators for registry different kind of classes with unique keys

- Register a environment: ``@registry.register_env``
- Register a trainer: ``@registry.register_trainer``
- Register a policy: ``@baseline_registry.register_policy``
"""

from typing import Optional

from habitat.core.registry import Registry


class BaselineRegistry(Registry):
    @classmethod
    def register_trainer(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL training algorithm to registry with key 'name'.

        Args:
            name: Key with which the trainer will be registered.
                If None will use the name of the class.

        """
        from habitat_baselines.common.base_trainer import BaseTrainer

        return cls._register_impl(
            "trainer", to_register, name, assert_type=BaseTrainer
        )

    @classmethod
    def get_trainer(cls, name):
        return cls._get_impl("trainer", name)

    @classmethod
    def register_env(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a environment to registry with key 'name'
            currently only support subclass of RLEnv.

        Args:
            name: Key with which the env will be registered.
                If None will use the name of the class.

        """
        from habitat import RLEnv

        return cls._register_impl("env", to_register, name, assert_type=RLEnv)

    @classmethod
    def get_env(cls, name):
        return cls._get_impl("env", name)
    
    @classmethod
    def register_policy(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a RL policy with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.rl.ppo.policy import Policy
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyPolicy(Policy):
                pass


            # or

            @baseline_registry.register_policy(name="MyPolicyName")
            class MyPolicy(Policy):
                pass

        """
        from habitat_baselines.rl.ppo.policy import Policy

        return cls._register_impl(
            "policy", to_register, name, assert_type=Policy
        )
    
    @classmethod
    def get_policy(cls, name: str):
        r"""Get the RL policy with :p:`name`."""
        return cls._get_impl("policy", name)
    
    @classmethod
    def register_obs_transformer(
        cls, to_register=None, *, name: Optional[str] = None
    ):
        r"""Register a Observation Transformer with :p:`name`.

        :param name: Key with which the policy will be registered.
            If :py:`None` will use the name of the class

        .. code:: py

            from habitat_baselines.common.obs_transformers import ObservationTransformer
            from habitat_baselines.common.baseline_registry import (
                baseline_registry
            )

            @baseline_registry.register_policy
            class MyObsTransformer(ObservationTransformer):
                pass


            # or

            @baseline_registry.register_policy(name="MyTransformer")
            class MyObsTransformer(ObservationTransformer):
                pass

        """
        from habitat_baselines.common.obs_transformers import (
            ObservationTransformer,
        )

        return cls._register_impl(
            "obs_transformer",
            to_register,
            name,
            assert_type=ObservationTransformer,
        )


baseline_registry = BaselineRegistry()
