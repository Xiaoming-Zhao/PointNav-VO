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
        # from habitat_baselines.common.base_trainer import BaseTrainer
        from pointnav_vo.rl.common.base_trainer import BaseTrainer

        return cls._register_impl("trainer", to_register, name, assert_type=BaseTrainer)

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
        r"""Register a RL policy to registry with key 'name'.

        Args:
            name: Key with which the policy will be registered.
                If None will use the name of the class.

        """
        # from habitat_baselines.common.base_trainer import BaseTrainer
        from pointnav_vo.rl.policies.policy import Policy

        return cls._register_impl("policy", to_register, name, assert_type=Policy)

    @classmethod
    def get_policy(cls, name):
        return cls._get_impl("policy", name)

    @classmethod
    def register_vo_model(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a Visual Odometry model to registry with key 'name'.

        Args:
            name: Key with which the VO model will be registered.
                If None will use the name of the class.

        """
        return cls._register_impl("vo_model", to_register, name,)

    @classmethod
    def get_vo_model(cls, name):
        return cls._get_impl("vo_model", name)

    @classmethod
    def register_vo_engine(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a Visual Odometry trainer to registry with key 'name'.

        Args:
            name: Key with which the VO model will be registered.
                If None will use the name of the class.

        """
        return cls._register_impl("vo_engine", to_register, name,)

    @classmethod
    def get_vo_engine(cls, name):
        return cls._get_impl("vo_engine", name)


baseline_registry = BaselineRegistry()
