#!/usr/bin/env python3

import argparse
import random
import numba
import os
import PIL
import numpy as np
from collections import OrderedDict
from gym.spaces import Discrete, Dict, Box

import torch

import habitat
from habitat import Config
from habitat.core.agent import Agent
from habitat_baselines.config.default import get_config

from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.rl.policies.resnet_policy import PointNavResNetPolicy
from pointnav_vo.vo.models.vo_cnn import VisualOdometryCNN
from pointnav_vo.utils.misc_utils import (
    batch_obs,
    Resizer,
    ResizeCenterCropper,
)
from pointnav_vo.utils.geometry_utils import (
    compute_goal_pos,
    pointgoal_polar2catesian,
    NormalizedDepth2TopDownViewHabitat,
)
from pointnav_vo.vo.common.common_vars import *


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class PointNavAgent(Agent):
    def __init__(self, config: Config, eval_server: str):
        self.config = config
        self._eval_server = eval_server

        spaces = {
            "pointgoal": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            "pointgoal_with_gps_compass": Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
        }

        spaces["depth"] = Box(
            low=0,
            high=1,
            shape=(
                config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH,
                1,
            ),
            dtype=np.float32,
        )
        spaces["rgb"] = Box(
            low=0,
            high=255,
            shape=(
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        observation_spaces = Dict(spaces)

        action_space = Discrete(len(config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))

        self.hidden_size = config.RL.PPO.hidden_size

        # set up nav policy
        self._set_up_nav_obs_transformer()
        self._setup_actor_critic_agent(observation_spaces, action_space)

        # set up vo module
        self._set_up_vo_obs_transformer()
        self._setup_vo_model()

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

        self.prev_pointgoal_relative_pos = None
        self.prev_obs = None

        self.episode_cnt = 0
        self.act_cnt = 0
        self.init_flag = True
        self.call_stop = False

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

    def _set_up_vo_obs_transformer(self) -> None:

        if self.config.VO.OBS_TRANSFORM == "resize_crop":
            self._vo_obs_transformer = ResizeCenterCropper(
                size=(self.config.VO.VIS_SIZE_W, self.config.VO.VIS_SIZE_H)
            )
        elif self.config.VO.OBS_TRANSFORM == "resize":
            self._vo_obs_transformer = Resizer(
                size=(self.config.VO.VIS_SIZE_W, self.config.VO.VIS_SIZE_H)
            )
        else:
            self._vo_obs_transformer = None

    def _setup_actor_critic_agent(self, observation_spaces, action_space) -> None:
        r"""Sets up actor critic and agent for DD-PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """
        # logger.add_filehandler(self.config.LOG_FILE)

        policy_cls = baseline_registry.get_policy(self.config.RL.Policy.name)
        assert policy_cls is not None, f"{self.config.RL.Policy.name} is not supported"

        normalize_visual_inputs_flag = (
            "rgb" in observation_spaces.spaces
            and "rgb" in self.config.RL.Policy.visual_types
        )
        self.actor_critic = policy_cls(
            observation_space=observation_spaces,
            action_space=action_space,
            hidden_size=self.config.RL.PPO.hidden_size,
            rnn_type=self.config.RL.Policy.rnn_backbone,
            num_recurrent_layers=self.config.RL.Policy.num_recurrent_layers,
            backbone=self.config.RL.Policy.visual_backbone,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            normalize_visual_inputs=normalize_visual_inputs_flag,
            obs_transform=self._nav_obs_transformer,
            vis_types=self.config.RL.Policy.visual_types,
        )
        self.actor_critic.to(self.device)

        print("\n", self.config.RL.Policy.pretrained_ckpt, "\n")

        pretrained_state = torch.load(
            self.config.RL.Policy.pretrained_ckpt, map_location=self.device
        )
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in pretrained_state["state_dict"].items()
            }
        )
        self.actor_critic.eval()

    def _setup_vo_model(self) -> None:

        # assert self.config.VO.REGRESS_MODEL.regress_type == "sep_act"

        assert (
            "rgb" in self.config.VO.REGRESS_MODEL.visual_type
            and "depth" in self.config.VO.REGRESS_MODEL.visual_type
        ), "Currently not support visual type other than RGB-D.\n"

        vo_model_cls = baseline_registry.get_vo_model(self.config.VO.REGRESS_MODEL.name)
        assert (
            vo_model_cls is not None
        ), f"{self.config.VO.REGRESS_MODEL.name} is not supported"

        agent_sensors = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        if "rgb" in self.config.VO.REGRESS_MODEL.visual_type:
            assert (
                "RGB_SENSOR" in agent_sensors
            ), f"Agent sensor {agent_sensors} does not contain RGB_SENSOR while vo model requries it."
        if "depth" in self.config.VO.REGRESS_MODEL.visual_type:
            assert (
                "DEPTH_SENSOR" in agent_sensors
            ), f"Agent sensor {agent_sensors} does not contain DEPTH_SENSOR while vo model requries it."

        if self.config.VO.REGRESS_MODEL.regress_type == "unified_act":
            output_dim = 3
            model_names = ["all"]
        elif self.config.VO.REGRESS_MODEL.regress_type == "sep_act":
            output_dim = 3
            model_names = list(ACT_IDX2NAME.values())
        else:
            raise ValueError

        self.vo_model = OrderedDict()
        for k in model_names:
            self.vo_model[k] = vo_model_cls(
                observation_space=self.config.VO.REGRESS_MODEL.visual_type,
                observation_size=(self.config.VO.VIS_SIZE_W, self.config.VO.VIS_SIZE_H),
                hidden_size=self.config.VO.REGRESS_MODEL.hidden_size,
                backbone=self.config.VO.REGRESS_MODEL.visual_backbone,
                normalize_visual_inputs="rgb"
                in self.config.VO.REGRESS_MODEL.visual_type,
                output_dim=output_dim,
                dropout_p=self.config.VO.REGRESS_MODEL.dropout_p,
                discretized_depth_channels=self.config.VO.REGRESS_MODEL.discretized_depth_channels,
            )
            self.vo_model[k].to(self.device)

        for k in model_names:
            print(
                "\n",
                self.config.VO.REGRESS_MODEL.all_pretrained_ckpt[
                    self.config.VO.REGRESS_MODEL.pretrained_type
                ][k],
                "\n",
            )
            pretrained_ckpt = torch.load(
                self.config.VO.REGRESS_MODEL.all_pretrained_ckpt[
                    self.config.VO.REGRESS_MODEL.pretrained_type
                ][k],
                map_location=self.device,
            )
            if "model_state" in pretrained_ckpt:
                self.vo_model[k].load_state_dict(pretrained_ckpt["model_state"])
            elif "model_states" in pretrained_ckpt:
                self.vo_model[k].load_state_dict(
                    pretrained_ckpt["model_states"][ACT_NAME2IDX[k]]
                )
            else:
                raise ValueError

            self.vo_model[k].eval()

            if "discretize_depth" in self.config.VO.REGRESS_MODEL.name:
                if self.config.VO.REGRESS_MODEL.discretize_depth in ["hard"]:
                    self._discretized_depth_end_vals = []
                    for i in np.arange(
                        self.config.VO.REGRESS_MODEL.discretized_depth_channels
                    ):
                        self._discretized_depth_end_vals.append(
                            i
                            * 1.0
                            / self.config.VO.REGRESS_MODEL.discretized_depth_channels
                        )
                    self._discretized_depth_end_vals.append(1.0)

            if "top_down" in self.config.VO.REGRESS_MODEL.name:
                top_down_view_infos = {
                    "min_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
                    "max_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                    "vis_size_h": self.config.VO.VIS_SIZE_H,
                    "vis_size_w": self.config.VO.VIS_SIZE_W,
                    "hfov_rad": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV,
                }
                self._top_down_view_generator = NormalizedDepth2TopDownViewHabitat(
                    **top_down_view_infos
                )

    def _discretize_depth_func(self, raw_depth):
        assert torch.max(raw_depth) <= 1.0
        assert torch.min(raw_depth) >= 0.0

        discretized_depth = torch.zeros(
            (*raw_depth.shape, self.config.VO.REGRESS_MODEL.discretized_depth_channels)
        )

        for i in np.arange(self.config.VO.REGRESS_MODEL.discretized_depth_channels):
            if i == self.config.VO.REGRESS_MODEL.discretized_depth_channels - 1:
                # include the last end values
                pos = torch.where(
                    (raw_depth >= self._discretized_depth_end_vals[i])
                    & (raw_depth <= self._discretized_depth_end_vals[i + 1])
                )
            else:
                pos = torch.where(
                    (raw_depth >= self._discretized_depth_end_vals[i])
                    & (raw_depth < self._discretized_depth_end_vals[i + 1])
                )

            if torch.numel(pos[0]) != 0:
                if self.config.VO.REGRESS_MODEL.discretize_depth == "hard":
                    discretized_depth[pos[0], pos[1], i] = 1.0
                else:
                    raise ValueError

        if self.config.VO.REGRESS_MODEL.discretize_depth == "hard":
            assert torch.sum(discretized_depth) == torch.numel(raw_depth)

        discretized_depth = discretized_depth.to(raw_depth.device)

        return discretized_depth

    def _compute_local_delta_states_from_vo(self, prev_obs, cur_obs, act):
        prev_rgb = prev_obs["rgb"]
        cur_rgb = cur_obs["rgb"]
        # [1, vis_size, vis_size, 6]
        rgb_pair = (
            torch.FloatTensor(np.concatenate([prev_rgb, cur_rgb], axis=2))
            .to(self.device)
            .unsqueeze(0)
        )

        # [vis_size, vis_size, 1]
        prev_depth = prev_obs["depth"]
        cur_depth = cur_obs["depth"]
        # [1, vis_size, vis_size, 2]
        depth_pair = (
            torch.FloatTensor(np.concatenate([prev_depth, cur_depth], axis=2))
            .to(self.device)
            .unsqueeze(0)
        )

        if self._vo_obs_transformer is not None:
            tmp_obs = torch.cat((rgb_pair, depth_pair), dim=3)

            if not self._vo_obs_transformer.channels_last:
                tmp_obs = tmp_obs.permute(0, 3, 1, 2)

            tmp_obs = self._vo_obs_transformer(tmp_obs)

            if not self._vo_obs_transformer.channels_last:
                tmp_obs = tmp_obs.permute(0, 2, 3, 1)

            rgb_pair = tmp_obs[:, :, :, :6]
            depth_pair = tmp_obs[:, :, :, 6:]

        obs_pairs = {"rgb": rgb_pair, "depth": depth_pair}

        if "discretize_depth" in self.config.VO.REGRESS_MODEL.name:
            assert depth_pair.size(-1) == 2
            prev_discretized_depth = self._discretize_depth_func(depth_pair[0, :, :, 0])
            cur_discretized_depth = self._discretize_depth_func(depth_pair[0, :, :, 1])
            discretized_depth_pair = torch.cat(
                (prev_discretized_depth, cur_discretized_depth), axis=2,
            ).unsqueeze(0)
            obs_pairs["discretized_depth"] = discretized_depth_pair

        if "top_down" in self.config.VO.REGRESS_MODEL.name:
            prev_top_down_view = self._top_down_view_generator.gen_top_down_view(
                depth_pair[0, :, :, 0].unsqueeze(-1)
            )
            cur_top_down_view = self._top_down_view_generator.gen_top_down_view(
                depth_pair[0, :, :, 1].unsqueeze(-1)
            )
            top_down_view_pair = torch.cat(
                (prev_top_down_view, cur_top_down_view), dim=2,
            ).unsqueeze(0)
            obs_pairs["top_down_view"] = top_down_view_pair

        local_delta_states = []

        with torch.no_grad():
            if self.config.VO.REGRESS_MODEL.regress_type == "unified_act":
                tmp_key = "all"
            else:
                tmp_key = ACT_IDX2NAME[act]
            if self.config.VO.REGRESS_MODEL.mode == "det":
                self.vo_model[tmp_key].eval()
                if "act_embed" in self.config.VO.REGRESS_MODEL.name:
                    actions = torch.Tensor([act]).long().to(rgb_pair.device)
                    tmp_deltas = self.vo_model[tmp_key](obs_pairs, actions)
                else:
                    tmp_deltas = self.vo_model[tmp_key](obs_pairs)
                local_delta_states = list(tmp_deltas.cpu().numpy()[0, :])
            elif self.config.VO.REGRESS_MODEL.mode == "rnd":
                self.vo_model[tmp_key].train()
                tmp_all_local_delta_states = []
                for tmp_i in range(self.config.VO.REGRESS_MODEL.rnd_mode_n):
                    tmp_deltas = self.vo_model[tmp_key](obs_pairs)
                    tmp_all_local_delta_states.append(tmp_deltas.cpu().numpy()[0, :])
                local_delta_states = list(
                    np.mean(np.array(tmp_all_local_delta_states), axis=0)
                )
            else:
                raise ValueError
        return local_delta_states

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        # for visual odometry module
        self.prev_pointgoal_relative_pos = None
        self.prev_obs = None

        print(f"Complete {self.episode_cnt} episodes.")
        self.episode_cnt += 1
        self.act_cnt = 0

        self.init_flag = True
        self.call_stop = False

    def act(self, observations):

        if self.init_flag:
            self.init_flag = False

        with torch.no_grad():

            if self.call_stop:
                action = STOP
            else:
                if self.prev_pointgoal_relative_pos is None:
                    # this is the first step of an episode
                    cur_pointgoal_relative_pos = {
                        "polar": observations["pointgoal"],
                        "cartesian": pointgoal_polar2catesian(
                            observations["pointgoal"]
                        ),
                    }
                else:
                    local_delta_states = self._compute_local_delta_states_from_vo(
                        self.prev_obs, observations, self.prev_actions[0, 0].item()
                    )

                    cur_pointgoal_relative_pos = compute_goal_pos(
                        self.prev_pointgoal_relative_pos["cartesian"],
                        local_delta_states,
                    )

                # self.prev_obs = cur_obs_dict
                self.prev_obs = observations
                self.prev_pointgoal_relative_pos = cur_pointgoal_relative_pos

                observations["pointgoal_with_gps_compass"] = cur_pointgoal_relative_pos[
                    "polar"
                ]

                batch = batch_obs([observations], device=self.device)

                self.actor_critic.eval()
                _, action, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    self.test_recurrent_hidden_states,
                    self.prev_actions,
                    self.not_done_masks,
                    # deterministic=False,
                    deterministic=True,
                )
                #  Make masks not done till reset (end of episode) will be called
                self.not_done_masks.fill_(1.0)
                self.prev_actions.copy_(action)

                action = action.item()

                if action == STOP:
                    # action = MOVE_FORWARD
                    action = STOP
                    self.call_stop = True

        return action


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]

    config = get_config(
        "challenge2020_pointnav_config.yaml", ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()

    print(config)

    config.defrost()
    config.TORCH_GPU_ID = 0
    config.SEED = 100
    config.TASK_CONFIG.SEED = 100
    config.TASK_CONFIG.SIMULATOR.SEED = 100
    config.freeze()

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    _seed_numba(config.SEED)
    torch.random.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    agent = PointNavAgent(config, args.evaluation)

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
        challenge._env.seed(config.SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    challenge.submit(agent)


if __name__ == "__main__":
    main()
