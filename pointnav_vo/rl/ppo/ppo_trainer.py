#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import tqdm
import cv2
import copy
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

import habitat
from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    angle_between_quaternions,
)
from habitat.utils.visualizations import maps

from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.utils.tensorboard_utils import TensorboardWriter
from pointnav_vo.utils.misc_utils import (
    batch_obs,
    generate_video,
    linear_decay,
)
from pointnav_vo.rl.common.base_trainer_with_vo import BaseRLTrainerWithVO
from pointnav_vo.rl.common.env_utils import construct_envs
from pointnav_vo.rl.common.environments import get_env_class
from pointnav_vo.rl.common.rollout_storage import RolloutStorage
from pointnav_vo.rl.ppo.ppo import PPO
from pointnav_vo.rl.ppo.policy import PointNavBaselinePolicy
from pointnav_vo.utils.geometry_utils import compute_goal_pos, compute_global_state
from pointnav_vo.vo.common.common_vars import *


@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainerWithVO):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        if config is not None:
            if "RANK" not in os.environ or (
                "RANK" in os.environ and int(os.environ.get("RANK", 0)) == 0
            ):
                logger.info(f"Trainer config:\n{self.config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = PointNavBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
        )
        self.actor_critic.to(self.device)

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        forward_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        # pth_time += time.time() - t_sample_action
        forward_time += time.time() - t_sample_action
        pth_time += forward_time

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        if self.config.RL.TUNE_WITH_VO:
            # update point goal's relative position with VO
            for k in self.vo_model:
                if self.config.VO.VO_TYPE == "REGRESS":
                    self.vo_model[k].eval()

            for i in range(len(observations)):
                if dones[i]:
                    # episode ends
                    cur_episode = self.envs.current_episodes()[i]
                    dx, _, dz = cur_episode.start_position
                    dyaw = 2 * np.arctan2(
                        cur_episode.start_rotation[1], cur_episode.start_rotation[3]
                    )
                    tmp_goal = compute_goal_pos(
                        np.array(cur_episode.goals[0].position), [dx, dz, dyaw]
                    )
                else:
                    # episode continues
                    (
                        local_delta_states,
                        local_delta_states_std,
                        extra_infos,
                    ) = self._compute_local_delta_states_from_vo(
                        self._prev_obs[i], observations[i], actions[i].cpu().item(),
                    )
                    tmp_goal = compute_goal_pos(
                        self._prev_goal_positions[i]["cartesian"], local_delta_states
                    )

                observations[i]["pointgoal_with_gps_compass"] = tmp_goal["polar"]
                self._prev_goal_positions[i] = tmp_goal

            self._prev_obs = observations

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, forward_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        forward_time = 0
        agent_update_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_forward_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats,
                    )
                    pth_time += delta_pth_time
                    forward_time += delta_forward_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time
                agent_update_time += delta_pth_time

                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    writer.add_scalars("metrics", metrics, count_steps)

                losses = [value_loss, action_loss]

                writer.add_scalars(
                    "losses",
                    {k: l for l, k in zip(losses, ["value", "policy"])},
                    count_steps,
                )

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )
                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "forward-time: {:.3f}\tagnet-update-time: {:.3f}\t"
                        "frames: {}".format(
                            update,
                            env_time,
                            pth_time,
                            forward_time,
                            agent_update_time,
                            count_steps,
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", dict(step=count_steps)
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.TOP_DOWN_MAP.TYPE = "ModifiedTopDownMap"
        config.TASK_CONFIG.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = (
            self.config.EVAL.DRAW_SHORTEST_PATH
        )
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.freeze()

        if checkpoint_index == 0:
            logger.info(f"Eval config:\n{config}\n")

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))

        # set up nav policy
        self._set_up_nav_obs_transformer()
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(f"Load policy weights from {checkpoint_path}\n.")
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        # set up vo model
        if self.config.VO.USE_VO_MODEL:
            self._set_up_vo_obs_transformer()
            self._setup_vo_model(config)

        # get name of performance metric, e.g. "spl"
        metric_name = self.config.TASK_CONFIG.TASK.MEASUREMENTS[0]
        metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
        measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
        assert measure_type is not None, "invalid measurement type {}".format(
            metric_cfg.TYPE
        )
        self.metric_uuid = measure_type(sim=None, task=None, config=None)._get_uuid()

        observations = self.envs.reset()

        # completely stuck count
        dx_stuck_cnt = [0 for _ in range(self.envs.num_envs)]
        dz_stuck_cnt = [0 for _ in range(self.envs.num_envs)]
        both_stuck_cnt = [0 for _ in range(self.envs.num_envs)]

        # env timing
        env_timings = []

        if self.config.VO.USE_VO_MODEL:
            # for computing goal position
            prev_obs = []
            prev_goal_positions = []
            # for computing VO L2 loss
            vo_l2_losses = []
            completed_vo_l2_losses = []
            all_vo_l2_losses = []
            # for computing VO pred std
            vo_pred_stds = []
            all_vo_pred_stds = []
            # for computing global state diff between VO and sim
            prev_global_state_from_vo = []
            diff_between_global_states = []
            all_diff_between_global_states = []
            # timing
            vo_timings = []
            all_vo_timings = []

            cur_episodes = self.envs.current_episodes()
            for i, tmp in enumerate(cur_episodes):
                dx, _, dz = tmp.start_position
                dyaw = 2 * np.arctan2(tmp.start_rotation[1], tmp.start_rotation[3])
                tmp_goal = compute_goal_pos(
                    np.array(tmp.goals[0].position), [dx, dz, dyaw]
                )
                observations[i]["pointgoal_with_gps_compass"] = tmp_goal["polar"]

                prev_goal_positions.append(tmp_goal)
                tmp_obs_dict = {"new_traj": True}
                for vis_type in ["rgb", "depth"]:
                    tmp_obs_dict[vis_type] = observations[i][vis_type]
                prev_obs.append(tmp_obs_dict)
                # for VO L2 loss
                vo_l2_losses.append([])
                completed_vo_l2_losses.append([])
                # for VO pred std
                vo_pred_stds.append([])
                # completely stuck count
                dx_stuck_cnt.append(0)
                dz_stuck_cnt.append(0)
                both_stuck_cnt.append(0)
                # for global state diff
                prev_global_state_from_vo.append(
                    (quaternion_from_coeff(tmp.start_rotation), tmp.start_position)
                )
                diff_between_global_states.append(())
                # timing
                vo_timings.append([])

        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(self.config.NUM_PROCESSES, 1, device=self.device)
        stats_episodes = dict()  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_PROCESSES)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.EVAL.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        # record the detailed information
        eval_scene_list = []
        eval_episode_info_dict = {"config": config}
        eval_traj_infos = [[] for _ in range(self.envs.num_envs)]

        self.actor_critic.eval()
        if self.config.VO.USE_VO_MODEL:
            if self.config.VO.VO_TYPE == "REGRESS":
                for k in self.vo_model:
                    self.vo_model[k].eval()

        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:

            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                self.actor_critic.eval()
                (_, actions, _, test_recurrent_hidden_states,) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

                prev_actions.copy_(actions)

            tmp_env_time = time.time()

            outputs = self.envs.step([a[0].item() for a in actions])

            env_timings.append(time.time() - tmp_env_time)

            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # add detailed trajectory info
            for i in range(self.envs.num_envs):
                if len(eval_traj_infos[i]) == 0:
                    # we add map info no matter whether the episode ends or not
                    eval_traj_infos[i].append(
                        infos[i]["top_down_map"]["extra_infos"]["map_infos"]
                    )

                # fmt: off

                if not_done_masks[i].item() != 0:
                    # episode continues
                    eval_traj_infos[i].append(
                        {
                            "action": actions[i][0].item(),
                            "prev_agent_state": infos[i]["top_down_map"]["extra_infos"]["prev_state"],
                            "cur_agent_state": infos[i]["top_down_map"]["extra_infos"]["cur_state"],
                            "prev_agent_angle": infos[i]["top_down_map"]["extra_infos"]["prev_agent_angle"],
                            "cur_agent_angle": infos[i]["top_down_map"]["agent_angle"],
                            "gt_delta": infos[i]["top_down_map"]["extra_infos"]["delta"],
                            "collision": int(infos[i]["collisions"]["is_collision"]),
                        }
                    )

                    if len(self.config.VIDEO_OPTION) > 0:
                        eval_traj_infos[i][-1]["fog_of_war_mask"] = infos[i]["top_down_map"]["fog_of_war_mask"]
                        eval_traj_infos[i][-1]["rgb"] = observations[i]["rgb"]
                        eval_traj_infos[i][-1]["depth"] = observations[i]["depth"]

                # fmt: on

            # visual odometry
            if self.config.VO.USE_VO_MODEL:
                for i in range(self.envs.num_envs):
                    if not_done_masks[i].item() == 0:
                        # episode ends
                        cur_episode = self.envs.current_episodes()[i]
                        dx, _, dz = cur_episode.start_position
                        dyaw = 2 * np.arctan2(
                            cur_episode.start_rotation[1], cur_episode.start_rotation[3]
                        )
                        # computing goal position
                        tmp_goal = compute_goal_pos(
                            np.array(cur_episode.goals[0].position), [dx, dz, dyaw]
                        )
                        # global state from VO
                        prev_global_state_from_vo[i] = (
                            quaternion_from_coeff(cur_episode.start_rotation),
                            cur_episode.start_position,
                        )

                        if len(eval_traj_infos[i]) > 1:
                            # VO L2 loss
                            completed_vo_l2_losses[i] = np.mean(
                                np.array(vo_l2_losses[i]), axis=0
                            )
                            all_vo_l2_losses.append(
                                (
                                    len(vo_l2_losses[i]),
                                    np.sum(np.array(vo_l2_losses[i]), axis=0),
                                )
                            )
                            vo_l2_losses[i] = []
                            # VO pred std
                            all_vo_pred_stds.append(
                                (
                                    len(vo_pred_stds[i]),
                                    np.sum(np.array(vo_pred_stds[i]), axis=0),
                                )
                            )
                            vo_pred_stds[i] = []

                            # compute diff between global state between VO and sim
                            state_from_sim = eval_traj_infos[i][-1]["cur_agent_state"]
                            state_from_vo = eval_traj_infos[i][-1][
                                "cur_agent_state_from_vo"
                            ]
                            diff_pos = np.abs(
                                np.array(state_from_sim["position"])
                                - np.array(state_from_vo["position"])
                            )
                            diff_yaw = angle_between_quaternions(
                                state_from_sim["rotation"], state_from_vo["rotation"]
                            )
                            # [diff_x, diff_z, diff_yaw]
                            diff_between_global_states[i] = [
                                diff_pos[0],
                                diff_pos[2],
                                diff_yaw,
                            ]
                            all_diff_between_global_states.append(
                                [diff_pos[0], diff_pos[2], diff_yaw]
                            )
                            # timing
                            all_vo_timings.append(
                                (len(vo_timings[i]), np.sum(vo_timings[i]),)
                            )
                            vo_timings[i] = []
                        else:
                            # episode ends after first step
                            assert (
                                len(vo_l2_losses[i]) == 0
                            ), f"\nvo_l2_losses: {i}, {vo_l2_losses[i]}\n"

                            all_vo_l2_losses.append((0, (0.0, 0.0, 0.0)))
                            vo_l2_losses[i] = []

                            all_vo_pred_stds.append((0, (0.0, 0.0, 0.0)))
                            vo_pred_stds[i] = []

                            diff_between_global_states[i] = [0.0, 0.0, 0.0]
                            all_diff_between_global_states.append([0.0, 0.0, 0.0])

                            all_vo_timings.append((0, 0.0,))
                            vo_timings[i] = []

                        # NOTE: no need for this step.
                        # Since we explicitly ensure that hidden states are zero-valued for every episode.
                        # test_recurrent_hidden_states[:, i, :] = torch.zeros(
                        #     self.actor_critic.net.num_recurrent_layers,
                        #     ppo_cfg.hidden_size,
                        #     device=self.device,
                        # )
                    else:
                        # episode continues
                        tmp_gt = np.array(
                            infos[i]["top_down_map"]["extra_infos"]["delta"]
                        )

                        # compute local delta states from VO

                        # NOTE: do not directly modify observations, which will cause bug when dealing envs_to_pause
                        # cur_obs = observations[i]

                        cur_obs = {"new_traj": False}
                        for obs_type in prev_obs[i].keys():
                            if obs_type != "new_traj":
                                cur_obs[obs_type] = observations[i][obs_type]

                        tmp_start = time.time()

                        (
                            local_delta_states,
                            local_delta_states_std,
                            extra_infos,
                        ) = self._compute_local_delta_states_from_vo(
                            prev_obs[i],
                            cur_obs,
                            actions[i][0].item(),
                            vis_video=len(self.config.VIDEO_OPTION) > 0,
                        )

                        vo_timings[i].append(time.time() - tmp_start)

                        vo_l2_losses[i].append(
                            np.abs(np.array(local_delta_states) - tmp_gt)
                        )
                        vo_pred_stds[i].append(local_delta_states_std)

                        if len(self.config.VIDEO_OPTION) > 0:
                            if "ego_top_down_map" in extra_infos:
                                eval_traj_infos[i][-1][
                                    "ego_top_down_map"
                                ] = extra_infos["ego_top_down_map"]

                        # add detailed trajectory info
                        # local state
                        eval_traj_infos[i][-1]["pred_delta"] = local_delta_states
                        # global state from VO
                        eval_traj_infos[i][-1]["prev_agent_state_from_vo"] = {
                            "rotation": prev_global_state_from_vo[i][0],
                            "position": prev_global_state_from_vo[i][1],
                        }
                        cur_agent_state_from_vo = compute_global_state(
                            prev_global_state_from_vo[i], local_delta_states
                        )
                        eval_traj_infos[i][-1]["cur_agent_state_from_vo"] = {
                            "rotation": cur_agent_state_from_vo[0],
                            "position": cur_agent_state_from_vo[1],
                        }
                        prev_global_state_from_vo[i] = cur_agent_state_from_vo
                        # VO pred var
                        eval_traj_infos[i][-1]["vo_pred_std"] = local_delta_states_std

                        # computing goal position
                        tmp_goal = compute_goal_pos(
                            prev_goal_positions[i]["cartesian"], local_delta_states
                        )

                        # NOTE: DEBUG
                        # print("\n", actions[i][0].item(), eval_traj_infos[i][-1]["gt_delta"], local_delta_states)

                    observations[i]["pointgoal_with_gps_compass"] = tmp_goal["polar"]
                    assert "new_traj" not in observations[i]

                    tmp_obs_dict = {"new_traj": dones[i] == 1}
                    for obs_type in prev_obs[i].keys():
                        if obs_type != "new_traj":
                            tmp_obs_dict[obs_type] = observations[i][obs_type]
                    prev_obs[i] = tmp_obs_dict
                    prev_goal_positions[i] = tmp_goal

            # NOTE: must put this code after VO processing
            batch = batch_obs(observations, self.device)

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs

            for i in range(n_envs):

                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    # episode ended
                    cur_scene_id = current_episodes[i].scene_id
                    cur_episode_id = current_episodes[i].episode_id

                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(self._extract_scalars_from_info(infos[i]))
                    episode_stats["len"] = len(eval_traj_infos[i]) - 1
                    current_episode_reward[i] = 0

                    # completely stuck cnt
                    episode_stats["dx_stuck"] = dx_stuck_cnt[i]
                    dx_stuck_cnt[i] = 0
                    episode_stats["dz_stuck"] = dz_stuck_cnt[i]
                    dz_stuck_cnt[i] = 0
                    episode_stats["both_stuck"] = both_stuck_cnt[i]
                    both_stuck_cnt[i] = 0

                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(cur_scene_id, cur_episode_id,)] = episode_stats

                    # NOTE: save episode level information
                    # since we place this code within the if branch for episode-end,
                    # we do not need to tackle the paused envrionment situation
                    eval_scene_list.append(cur_scene_id)

                    if cur_scene_id not in eval_episode_info_dict:
                        eval_episode_info_dict[cur_scene_id] = {}
                    eval_episode_info_dict[cur_scene_id][cur_episode_id] = {}

                    eval_episode_info_dict[cur_scene_id][cur_episode_id]["start"] = {
                        "position": current_episodes[i].start_position,
                        "rotation": current_episodes[i].start_rotation,
                    }

                    eval_episode_info_dict[cur_scene_id][cur_episode_id]["goal"] = {
                        "position": current_episodes[i].goals[0].position,
                    }

                    eval_episode_info_dict[cur_scene_id][cur_episode_id]["stat"] = {
                        tmp_k: tmp_v for tmp_k, tmp_v in episode_stats.items()
                    }
                    eval_episode_info_dict[cur_scene_id][cur_episode_id][
                        "map"
                    ] = eval_traj_infos[i][0]
                    eval_episode_info_dict[cur_scene_id][cur_episode_id][
                        "traj"
                    ] = eval_traj_infos[i][1:]
                    eval_traj_infos[i] = []

                    if self.config.VO.USE_VO_MODEL:
                        eval_episode_info_dict[cur_scene_id][cur_episode_id][
                            "vo_l2_loss"
                        ] = completed_vo_l2_losses[i]
                        eval_episode_info_dict[cur_scene_id][cur_episode_id][
                            "diff_between_global_states"
                        ] = diff_between_global_states[i]

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )
                        rgb_frames[i] = []
                else:
                    # episode continues
                    if int(infos[i]["collisions"]["is_collision"]) == 1:
                        # completely stuck
                        dx_stuck_flag = False
                        dz_stuck_flag = False
                        if infos[i]["top_down_map"]["extra_infos"]["delta"][0] == 0.0:
                            # check delta_x
                            dx_stuck_cnt[i] += 1
                            dx_stuck_flag = True
                        if infos[i]["top_down_map"]["extra_infos"]["delta"][1] == 0.0:
                            # check delta_z
                            dz_stuck_cnt[i] += 1
                            dz_stuck_flag = True
                        if dx_stuck_flag and dz_stuck_flag:
                            both_stuck_cnt[i] += 1

                    if len(self.config.VIDEO_OPTION) > 0:
                        frame = observations_to_image(observations[i], infos[i])
                        rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # NOTE: tackle pasued environment situation
            eval_traj_infos = [
                _ for i, _ in enumerate(eval_traj_infos) if i not in envs_to_pause
            ]
            if self.config.VO.USE_VO_MODEL:
                # for computing goal position
                prev_goal_positions = [
                    _
                    for i, _ in enumerate(prev_goal_positions)
                    if i not in envs_to_pause
                ]
                prev_obs = [_ for i, _ in enumerate(prev_obs) if i not in envs_to_pause]
                # for computing VO L2 loss
                vo_l2_losses = [
                    _ for i, _ in enumerate(vo_l2_losses) if i not in envs_to_pause
                ]
                completed_vo_l2_losses = [
                    _
                    for i, _ in enumerate(completed_vo_l2_losses)
                    if i not in envs_to_pause
                ]
                # for timing
                vo_timings = [
                    _ for i, _ in enumerate(vo_timings) if i not in envs_to_pause
                ]
                # for computing VO pred std
                vo_pred_stds = [
                    _ for i, _ in enumerate(vo_pred_stds) if i not in envs_to_pause
                ]
                # for stuck count
                dx_stuck_cnt = [
                    _ for i, _ in enumerate(dx_stuck_cnt) if i not in envs_to_pause
                ]
                dz_stuck_cnt = [
                    _ for i, _ in enumerate(dz_stuck_cnt) if i not in envs_to_pause
                ]
                both_stuck_cnt = [
                    _ for i, _ in enumerate(both_stuck_cnt) if i not in envs_to_pause
                ]
                # for computing global state diff between VO and sim
                prev_global_state_from_vo = [
                    _
                    for i, _ in enumerate(prev_global_state_from_vo)
                    if i not in envs_to_pause
                ]
                diff_between_global_states = [
                    _
                    for i, _ in enumerate(diff_between_global_states)
                    if i not in envs_to_pause
                ]

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()]) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        # NOTE
        eval_scene_list = [os.path.basename(_) for _ in eval_scene_list]
        logger.info(f"\nAll scenes: {set(eval_scene_list)}\n")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalars(
            "eval_reward", {"average reward": aggregated_stats["reward"]}, step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        if len(metrics) > 0:
            writer.add_scalars("eval_metrics", metrics, step_id)

        self.envs.close()
        aggregated_stats["env_timing"] = np.sum(env_timings) / len(env_timings)
        logger.info(f"Average env timing per step: {aggregated_stats['env_timing']}")

        if self.config.VO.USE_VO_MODEL:
            # vo L2 loss
            all_vo_l2_losses = list(zip(*all_vo_l2_losses))
            all_steps = np.sum(all_vo_l2_losses[0])
            loss_sum = np.sum(np.array(all_vo_l2_losses[1]), axis=0)
            aggregated_stats["vo_l2_loss"] = loss_sum / all_steps
            logger.info(f"Average vo l2 loss per step: {loss_sum / all_steps}")
            # vo timings
            all_vo_timings = list(zip(*all_vo_timings))
            timing_sum = np.sum(all_vo_timings[1])
            aggregated_stats["vo_timing"] = timing_sum / np.sum(
                all_vo_timings[0]
            )  # all_steps
            logger.info(f"Average vo timing per step: {timing_sum / all_steps}")
            # vo pred vars
            all_vo_pred_stds = list(zip(*all_vo_pred_stds))
            pred_std_sum = np.sum(np.array(all_vo_pred_stds[1]), axis=0)
            aggregated_stats["vo_pred_std"] = pred_std_sum / all_steps
            logger.info(f"Average vo pred std per step: {pred_std_sum / all_steps}")
            # diff between global states
            avg_diff = np.mean(np.array(all_diff_between_global_states), axis=0)
            aggregated_stats["vo_l2_gloabl_state"] = avg_diff
            logger.info(
                f"Average episode global state diffs at the last step: {avg_diff}"
            )

        return (
            eval_episode_info_dict,
            {tmp_k: [tmp_v] for tmp_k, tmp_v in aggregated_stats.items()},
        )
