"""
Modifications are based on https://github.com/facebookresearch/habitat-api/blob/e0807ed403902de78ee37b36766c2f06e5886fea/habitat/core/env.py
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Type, Union
import gym

import habitat
from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.simulator import Observations


class ModifiedEnvForVis(habitat.Env):
    def reset_to_episode(self, target_episode: Episode) -> Observations:
        r"""Resets the environments and returns the initial observations.
        :return: initial observations from the environment.
        """
        self._reset_stats()

        assert len(self.episodes) > 0, "Episodes list is empty"
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        # self._current_episode = next(self._episode_iterator)
        self._current_episode = target_episode
        self.reconfigure(self._config)

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode, task=self.task
        )

        return observations


class ModifiedRLEnvForVis(habitat.RLEnv):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.
    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.
    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: ModifiedEnvForVis

    def __init__(self, config: Config, dataset: Optional[Dataset] = None) -> None:
        """Constructor
        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """

        self._env = ModifiedEnvForVis(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    def reset_to_episode(self, target_episode: Episode) -> Observations:
        return self._env.reset_to_episode(target_episode)


class SimpleRLEnvForVis(ModifiedRLEnvForVis):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
