#!/usr/bin/env python3

import contextlib
import os
import sys
import random
import time
import datetime
import joblib
import traceback
import numpy as np
from collections import OrderedDict, defaultdict, deque
from typing import Dict, List

import gym
from gym import spaces

try:
    from gym.spaces import Dict as SpaceDict
except:
    from gym.spaces.dict_space import Dict as SpaceDict

import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger

from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.utils.config_utils import update_config_log, convert_cfg_to_dict
from pointnav_vo.utils.geometry_utils import compute_goal_pos
from pointnav_vo.utils.tensorboard_utils import TensorboardWriter
from pointnav_vo.utils.misc_utils import (
    batch_obs,
    linear_decay,
    ResizeCenterCropper,
    Resizer,
)
from pointnav_vo.rl.common.env_utils import construct_envs
from pointnav_vo.rl.common.environments import get_env_class
from pointnav_vo.rl.common.rollout_storage import RolloutStorage
from pointnav_vo.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    requeue_job,
    save_interrupted_state,
)
from pointnav_vo.rl.ddppo.algo.ddppo import DDPPO
from pointnav_vo.rl.ppo.ppo_trainer import PPOTrainer
from pointnav_vo.vo.common.common_vars import *


@baseline_registry.register_trainer(name="efficient_ddppo")
class DDPPOTrainer(PPOTrainer):
    # DD-PPO cuts rollouts short to mitigate the straggler effect
    # This, in theory, can cause some rollouts to be very short.
    # All rollouts contributed equally to the loss/model-update,
    # thus very short rollouts can be problematic.  This threshold
    # limits the how short a short rollout can be as a fraction of the
    # max rollout length
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None, run_type="train", verbose=True):

        if config.RESUME_TRAIN:
            new_config = config

            self.resume_state_file = config.RESUME_STATE_FILE
            print(f"\nLoaded from: {self.resume_state_file}\n")
            interrupted_state = torch.load(config.RESUME_STATE_FILE, map_location="cpu")
            config = interrupted_state["config"]

            config.defrost()

            config.TASK_CONFIG.DATASET.update(new_config.TASK_CONFIG.DATASET)
            if "TUNE_WITH_VO" not in config.RL:
                config.RL.TUNE_WITH_VO = new_config.RL.TUNE_WITH_VO

            config.CHECKPOINT_INTERVAL = new_config.CHECKPOINT_INTERVAL

            config = update_config_log(config, run_type, new_config.LOG_DIR)

            config.freeze()
        else:
            self.resume_state_file = None

        super().__init__(config)

    def _set_up_nav_obs_transformer(self) -> None:

        if self.config.RL.OBS_TRANSFORM == "resize_crop":
            self._nav_obs_transformer = ResizeCenterCropper(
                size=(self.config.RL.VIS_SIZE_W, self.config.RL.VIS_SIZE_H)
            )
        elif self.config.RL.OBS_TRANSFORM == "resize":
            self._nav_obs_transformer = Resizer(
                size=(self.config.RL.VIS_SIZE_W, self.config.RL.VIS_SIZE_H)
            )
        else:
            self._nav_obs_transformer = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for DD-PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """

        policy_cls = baseline_registry.get_policy(self.config.RL.Policy.name)
        assert policy_cls is not None, f"{self.config.RL.Policy.name} is not supported"

        normalize_visual_inputs_flag = (
            "rgb" in self.envs.observation_spaces[0].spaces
            and "rgb" in self.config.RL.Policy.visual_types
        )
        self.actor_critic = policy_cls(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            rnn_type=self.config.RL.Policy.rnn_backbone,
            num_recurrent_layers=self.config.RL.Policy.num_recurrent_layers,
            backbone=self.config.RL.Policy.visual_backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs=normalize_visual_inputs_flag,
            obs_transform=self._nav_obs_transformer,
            vis_types=self.config.RL.Policy.visual_types,
        )
        self.actor_critic.to(self.device)

        if self.config.RL.DDPPO.pretrained_encoder or self.config.RL.DDPPO.pretrained:
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = DDPPO(
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

    def train(self) -> None:
        r"""Main method for DD-PPO.

        Returns:
            None
        """
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore("rollout_tracker", tcp_store)
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank

        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.NUM_PROCESSES
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

        ppo_cfg = self.config.RL.PPO

        self._set_up_nav_obs_transformer()
        self._setup_actor_critic_agent(ppo_cfg)
        self.agent.init_distributed(find_unused_params=True)

        if self.config.RL.TUNE_WITH_VO:
            self._set_up_vo_obs_transformer()
            self._setup_vo_model(self.config)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.agent.parameters()
                        if param.requires_grad
                    )
                )
            )

        observations = self.envs.reset()
        batch = batch_obs(observations)

        if self.config.RL.TUNE_WITH_VO:
            self._prev_obs = observations
            self._prev_goal_positions = []

            cur_episodes = self.envs.current_episodes()
            for i, tmp in enumerate(cur_episodes):
                dx, _, dz = tmp.start_position
                dyaw = 2 * np.arctan2(tmp.start_rotation[1], tmp.start_rotation[3])
                tmp_goal = compute_goal_pos(
                    np.array(tmp.goals[0].position), [dx, dz, dyaw]
                )
                observations[i]["pointgoal_with_gps_compass"] = tmp_goal["polar"]
                self._prev_goal_positions.append(tmp_goal)

        obs_space = self.envs.observation_spaces[0]
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = SpaceDict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
        )
        rollouts.to(self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1, device=self.device),
            reward=torch.zeros(self.envs.num_envs, 1, device=self.device),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        log_frames_list = []
        log_reward_list = []
        log_policy_loss_list = []
        log_value_loss_list = []
        log_metrics_dict = defaultdict(list)

        t_start = time.time()
        env_time = 0
        pth_time = 0
        forward_time = 0
        agent_update_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # interrupted_state = load_interrupted_state(self.resume_state_file)
        if self.resume_state_file is not None:
            # NOTE: must map to CPU, otherwise it will wait forever
            interrupted_state = torch.load(self.resume_state_file, map_location="cpu")
        else:
            interrupted_state = None

        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(interrupted_state["optim_state"])
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            forward_time = requeue_stats["forward_time"]
            agent_update_time = requeue_stats["agent_update_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs)
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:

            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():

                    print(f"\nStart exiting rank {self.world_rank}\n")
                    self.envs.close()

                    # if self.world_rank == 0:
                    if REQUEUE.is_set() and self.world_rank == 0:
                        print(f"\nStart saving rank {self.world_rank}\n")
                        # tmp_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            forward_time=forward_time,
                            agent_update_time=agent_update_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optim_state=self.agent.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        )
                        print(f"\nEnd saving rank {self.world_rank}\n")

                    print(f"\nEnd exiting rank {self.world_rank}\n")
                    requeue_job()
                    return

                count_steps_delta = 0
                self.agent.eval()
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
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.agent.train()
                if self._static_encoder:
                    self._encoder.eval()

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time
                agent_update_time += delta_pth_time

                stats_ordering = list(sorted(running_episode_stats.keys()))
                stats = torch.stack(
                    [running_episode_stats[k] for k in stats_ordering], 0
                )

                distrib.all_reduce(stats)

                for i, k in enumerate(stats_ordering):
                    window_episode_stats[k].append(stats[i].clone())

                stats = torch.tensor(
                    [value_loss, action_loss, count_steps_delta], device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[2].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    self._log_func(
                        writer,
                        update,
                        stats,
                        count_steps,
                        count_checkpoints,
                        window_episode_stats,
                        t_start,
                        prev_time,
                        env_time,
                        pth_time,
                        forward_time,
                        agent_update_time,
                        lr_scheduler,
                        log_frames_list,
                        log_reward_list,
                        log_policy_loss_list,
                        log_value_loss_list,
                        log_metrics_dict,
                    )

                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        count_checkpoints += 1
                        log_frames_list = []
                        log_reward_list = []
                        log_policy_loss_list = []
                        log_value_loss_list = []
                        log_metrics_dict = defaultdict(list)

            self.envs.close()

    def _log_func(
        self,
        writer,
        update,
        stats,
        count_steps,
        count_checkpoints,
        window_episode_stats,
        t_start,
        prev_time,
        env_time,
        pth_time,
        forward_time,
        agent_update_time,
        lr_scheduler,
        log_frames_list,
        log_reward_list,
        log_policy_loss_list,
        log_value_loss_list,
        log_metrics_dict,
    ):

        log_frames_list.append(count_steps)
        writer.add_scalar("Train/update", update, count_steps)

        losses = [
            stats[0].item() / self.world_size,
            stats[1].item() / self.world_size,
        ]

        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in window_episode_stats.items()
        }

        deltas["count"] = max(deltas["count"], 1.0)

        tmp_avg_ret = deltas["reward"] / deltas["count"]
        writer.add_scalar(
            "Train/reward", tmp_avg_ret, count_steps,
        )
        log_reward_list.append(tmp_avg_ret)

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        if len(metrics) > 0:
            for k in metrics:
                writer.add_scalar(f"Train/{k}", metrics[k], count_steps)
                log_metrics_dict[k].append(metrics[k])

        for tmp_i, param_group in enumerate(self.agent.optimizer.param_groups):
            writer.add_scalar(
                f"LR/group_{tmp_i}", param_group["lr"], count_steps,
            )

        writer.add_scalars(
            "Train/losses",
            {k: l for l, k in zip(losses, ["value", "policy"])},
            count_steps,
        )
        log_value_loss_list.append(losses[0])
        log_policy_loss_list.append(losses[1])

        # log stats
        if update > 0 and update % self.config.LOG_INTERVAL == 0:

            # tmp_fps = count_steps / ((time.time() - t_start) + prev_time)
            tmp_fps = count_steps / env_time
            logger.info("update: {}\tfps: {:.3f}\t".format(update, tmp_fps,))
            writer.add_scalar("Simulation/FPS", tmp_fps, count_steps)

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "forward-time: {:.3f}\tagent-update-time: {:.3f}\t"
                "frames: {}".format(
                    update,
                    env_time,
                    pth_time,
                    forward_time,
                    agent_update_time,
                    count_steps,
                )
            )
            # writer.add_scalar("Simulation/env-time", env_time, count_steps)
            # writer.add_scalar("Simulation/pth-time", pth_time, count_steps)
            # writer.add_scalar("Simulation/forward-time", forward_time, count_steps)
            # writer.add_scalar("Simulation/agent-update-time", agent_update_time, count_steps)
            # writer.add_scalar("Simulation/replay-buffer-time",
            #                   pth_time - forward_time - agent_update_time, count_steps)
            writer.add_scalars(
                "Simulation/All-Time",
                {"env-time": env_time, "pth-time": pth_time},
                count_steps,
            )
            writer.add_scalars(
                "Simulation/Pth-Time",
                {
                    "forward-time": forward_time,
                    "agent-update-time": agent_update_time,
                    "replay-buffer-time": pth_time - forward_time - agent_update_time,
                },
                count_steps,
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

        if update % self.config.CHECKPOINT_INTERVAL == 0:
            # save infos for future analysis
            tmp_save_dict = {
                "episode_ret": log_reward_list,
                "value_loss": log_value_loss_list,
                "policy_loss": log_policy_loss_list,
                "frames": log_frames_list,
            }
            for k in log_metrics_dict:
                tmp_save_dict[k] = log_metrics_dict[k]
            self._save_info_dict(
                tmp_save_dict, os.path.join(self.config.INFO_DIR, "train_infos.p")
            )

            # save checkpoint for model
            requeue_stats = dict(
                env_time=env_time,
                pth_time=pth_time,
                forward_time=forward_time,
                agent_update_time=agent_update_time,
                count_steps=count_steps,
                count_checkpoints=count_checkpoints,
                start_update=update,
                prev_time=(time.time() - t_start) + prev_time,
            )
            torch.save(
                dict(
                    state_dict=self.agent.state_dict(),
                    optim_state=self.agent.optimizer.state_dict(),
                    lr_sched_state=lr_scheduler.state_dict(),
                    config=self.config,
                    requeue_stats=requeue_stats,
                ),
                os.path.join(
                    self.config.CHECKPOINT_FOLDER,
                    "ckpt_{}.update_{}.frames_{}.pth".format(
                        count_checkpoints, update, int(count_steps)
                    ),
                ),
            )
