#!/usr/bin/env python3

import numpy as np
from typing import ClassVar, Dict, List
from collections import defaultdict, OrderedDict

import torch

import habitat
from habitat import Config, logger

from pointnav_vo.rl.common.base_trainer import BaseRLTrainer
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.utils.geometry_utils import (
    NormalizedDepth2TopDownViewHabitat,
    NormalizedDepth2TopDownViewHabitatTorch,
)
from pointnav_vo.utils.tensorboard_utils import TensorboardWriter
from pointnav_vo.utils.misc_utils import ResizeCenterCropper, Resizer
from pointnav_vo.vo.common.common_vars import *


class BaseRLTrainerWithVO(BaseRLTrainer):
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

    def _setup_vo_model(self, all_cfg: Config) -> None:

        if all_cfg.VO.VO_TYPE == "REGRESS":
            model_cls_name = all_cfg.VO.REGRESS_MODEL.name
            vo_model_cls = baseline_registry.get_vo_model(model_cls_name)
        else:
            raise NotImplementedError
        assert vo_model_cls is not None, f"{model_cls_name} is not supported"

        agent_sensors = all_cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
        if "rgb" in all_cfg.VO.REGRESS_MODEL.visual_type:
            assert (
                "RGB_SENSOR" in agent_sensors
            ), f"Agent sensor {agent_sensors} does not contain RGB_SENSOR while vo model requries it."
        if "depth" in all_cfg.VO.REGRESS_MODEL.visual_type:
            assert (
                "DEPTH_SENSOR" in agent_sensors
            ), f"Agent sensor {agent_sensors} does not contain DEPTH_SENSOR while vo model requries it."

        if all_cfg.VO.VO_TYPE == "REGRESS":
            if all_cfg.VO.REGRESS_MODEL.regress_type == "unified_act":
                output_dim = 3
                model_names = ["all"]
            elif all_cfg.VO.REGRESS_MODEL.regress_type == "sep_act":
                output_dim = 3
                model_names = [_ for _ in list(ACT_IDX2NAME.values()) if _ != "unified"]
            else:
                raise ValueError

            self.vo_model = OrderedDict()
            for k in model_names:
                self.vo_model[k] = vo_model_cls(
                    observation_space=all_cfg.VO.REGRESS_MODEL.visual_type,
                    observation_size=(
                        self.config.VO.VIS_SIZE_W,
                        self.config.VO.VIS_SIZE_H,
                    ),
                    hidden_size=all_cfg.VO.REGRESS_MODEL.hidden_size,
                    backbone=all_cfg.VO.REGRESS_MODEL.visual_backbone,
                    normalize_visual_inputs=True,  # "rgb" in all_cfg.VO.REGRESS_MODEL.visual_type,
                    output_dim=output_dim,
                    dropout_p=all_cfg.VO.REGRESS_MODEL.dropout_p,
                    discretized_depth_channels=self.config.VO.REGRESS_MODEL.discretized_depth_channels,
                )
                self.vo_model[k].to(self.device)

            if all_cfg.VO.REGRESS_MODEL.pretrained:
                for k in model_names:
                    logger.info(
                        f"Load pretrained weights of vo {k} from {all_cfg.VO.REGRESS_MODEL.pretrained_ckpt[k]}"
                    )
                    pretrained_ckpt = torch.load(
                        all_cfg.VO.REGRESS_MODEL.pretrained_ckpt[k]
                    )

                    if "model_state" in pretrained_ckpt:
                        self.vo_model[k].load_state_dict(pretrained_ckpt["model_state"])
                    elif "model_states" in pretrained_ckpt:
                        self.vo_model[k].load_state_dict(
                            pretrained_ckpt["model_states"][ACT_NAME2IDX[k]]
                        )
                    else:
                        raise ValueError

            if (
                "discretize_depth" in self.config.VO.REGRESS_MODEL.name
                or "dd" in self.config.VO.REGRESS_MODEL.name
            ):
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
                else:
                    raise NotImplementedError

            if "top_down" in self.config.VO.REGRESS_MODEL.name:
                top_down_view_infos = {
                    "min_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
                    "max_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                    "vis_size_h": self.config.VO.VIS_SIZE_H,
                    "vis_size_w": self.config.VO.VIS_SIZE_W,
                    "hfov_rad": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV,
                }
                self._top_down_view_generator = NormalizedDepth2TopDownViewHabitatTorch(
                    **top_down_view_infos
                )

            logger.info(f"Visual Odometry model:\n{list(self.vo_model.values())[0]}")
        else:
            raise ValueError("Incompatible choise of VO type.")

    def _discretize_depth_func(self, raw_depth):
        assert torch.max(raw_depth) <= 1.0
        assert torch.min(raw_depth) >= 0.0

        discretized_depth = torch.zeros(
            (*raw_depth.shape, self.config.VO.REGRESS_MODEL.discretized_depth_channels)
        ).to(raw_depth.device)

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
                    raise NotImplementedError

        if self.config.VO.REGRESS_MODEL.discretize_depth == "hard":
            assert torch.sum(discretized_depth) == torch.numel(raw_depth)

        discretized_depth = discretized_depth

        return discretized_depth

    def _compute_local_delta_states_from_vo(
        self, prev_obs, cur_obs, act, vis_video=False,
    ):
        prev_rgb = prev_obs["rgb"]
        cur_rgb = cur_obs["rgb"]

        rgb_pair = torch.cat(
            [
                torch.FloatTensor(prev_rgb).to(self.device),
                torch.FloatTensor(cur_rgb).to(self.device),
            ],
            dim=2,
        ).unsqueeze(0)

        # [vis_size, vis_size, 1]
        prev_depth = prev_obs["depth"]
        cur_depth = cur_obs["depth"]

        depth_pair = torch.cat(
            [
                torch.FloatTensor(prev_depth).to(self.device),
                torch.FloatTensor(cur_depth).to(self.device),
            ],
            dim=2,
        ).unsqueeze(0)

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

        if self.config.VO.VO_TYPE == "REGRESS":
            if (
                "discretize_depth" in self.config.VO.REGRESS_MODEL.name
                or "dd" in self.config.VO.REGRESS_MODEL.name
            ):

                # process depth discretization
                assert depth_pair.size(-1) == 2

                prev_discretized_depth = self._discretize_depth_func(
                    depth_pair[0, :, :, 0]
                )
                cur_discretized_depth = self._discretize_depth_func(
                    depth_pair[0, :, :, 1]
                )
                discretized_depth_pair = torch.cat(
                    (prev_discretized_depth, cur_discretized_depth), dim=2,
                ).unsqueeze(0)

                obs_pairs["discretized_depth"] = discretized_depth_pair

            if "top_down" in self.config.VO.REGRESS_MODEL.name:

                # process top-down projection
                if isinstance(
                    self._top_down_view_generator,
                    NormalizedDepth2TopDownViewHabitatTorch,
                ):
                    prev_top_down_view = self._top_down_view_generator.gen_top_down_view(
                        depth_pair[0, :, :, 0].unsqueeze(-1)
                    )
                    cur_top_down_view = self._top_down_view_generator.gen_top_down_view(
                        depth_pair[0, :, :, 1].unsqueeze(-1)
                    )
                    top_down_view_pair = torch.cat(
                        (prev_top_down_view, cur_top_down_view), dim=2,
                    ).unsqueeze(0)
                elif isinstance(
                    self._top_down_view_generator, NormalizedDepth2TopDownViewHabitat
                ):
                    prev_top_down_view = self._top_down_view_generator.gen_top_down_view(
                        depth_pair[0, :, :, 0, np.newaxis].cpu().numpy()
                    )
                    cur_top_down_view = self._top_down_view_generator.gen_top_down_view(
                        depth_pair[0, :, :, 1, np.newaxis].cpu().numpy()
                    )
                    top_down_view_pair = (
                        torch.FloatTensor(
                            np.concatenate(
                                [prev_top_down_view, cur_top_down_view], axis=2
                            )
                        )
                        .to(depth_pair.device)
                        .unsqueeze(0)
                    )
                else:
                    raise ValueError

                obs_pairs["top_down_view"] = top_down_view_pair

        local_delta_states = []
        local_delta_states_std = []
        extra_infos = {}
        if vis_video:
            extra_infos["ego_top_down_map"] = cur_top_down_view

        with torch.no_grad():
            if self.config.VO.VO_TYPE == "REGRESS":

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
                    local_delta_states = tmp_deltas.cpu().numpy()[0, :]
                    local_delta_states = list(local_delta_states)
                    local_delta_states_std = [0, 0, 0]
                elif self.config.VO.REGRESS_MODEL.mode == "rnd":
                    self.vo_model[tmp_key].train()
                    tmp_all_local_delta_states = []
                    for tmp_i in range(self.config.VO.REGRESS_MODEL.rnd_mode_n):
                        tmp_deltas = (
                            self.vo_model[tmp_key](obs_pairs).cpu().numpy()[0, :]
                        )
                        tmp_all_local_delta_states.append(tmp_deltas)
                    local_delta_states = list(
                        np.mean(np.array(tmp_all_local_delta_states), axis=0)
                    )
                    local_delta_states_std = list(
                        np.std(tmp_all_local_delta_states, axis=0)
                    )
                else:
                    pass
            else:
                raise NotImplementedError

        return local_delta_states, local_delta_states_std, extra_infos
