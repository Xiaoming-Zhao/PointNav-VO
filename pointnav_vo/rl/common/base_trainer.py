#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import joblib
import tqdm
import glob
import imageio
import copy
import numpy as np
from typing import ClassVar, Dict, List
from collections import defaultdict

import torch

import habitat
from habitat import Config, logger
from habitat.utils.visualizations import maps

from pointnav_vo.utils.tensorboard_utils import TensorboardWriter
from pointnav_vo.vis.utils import resize_top_down_map
from pointnav_vo.vo.common.common_vars import *


EPSILON = 1e-8


class BaseTrainer:
    r"""Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer, SLAM or imitation learner.
    Includes only the most basic functionality.
    """

    supported_tasks: ClassVar[List[str]]

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError


class BaseRLTrainer(BaseTrainer):
    r"""Base trainer class for RL trainers. Future RL-specific
    methods should be hosted here.
    """
    device: torch.device
    config: Config
    video_option: List[str]
    _flush_secs: int

    def __init__(self, config: Config):
        super().__init__()
        assert config is not None, "needs config file to initialize trainer"
        self.config = config
        self._flush_secs = 30

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        r"""Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"

        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:

            if os.path.isfile(self.config.EVAL.EVAL_CKPT_PATH):
                # evaluate singe checkpoint
                eval_f_list = [self.config.EVAL.EVAL_CKPT_PATH]
            else:
                # evaluate multiple checkpoints in order
                eval_f_list = list(
                    glob.glob(os.path.join(self.config.EVAL.EVAL_CKPT_PATH, "*.pth"))
                )
                eval_f_list = sorted(eval_f_list, key=lambda x: os.stat(x).st_mtime)

            for ckpt_id, current_ckpt in tqdm.tqdm(enumerate(eval_f_list)):
                logger.info(f"======= current_ckpt: {current_ckpt} =======\n")

                (
                    current_episode_result,
                    current_overall_result,
                ) = self._eval_checkpoint(
                    checkpoint_path=current_ckpt,
                    writer=writer,
                    checkpoint_index=ckpt_id,
                )

                try:
                    # assume the file name is ckpt_XX.update_XX.frames_XX.pth
                    current_ckpt_filename = os.path.basename(current_ckpt)
                    current_frame = int(
                        current_ckpt_filename.split("frames_")[1].split(".")[0]
                    )
                    current_overall_result["frames"] = [current_frame]
                except:
                    current_overall_result["frames"] = [None]

                self._save_info_dict(
                    current_episode_result,
                    os.path.join(
                        self.config.INFO_DIR,
                        "{}.infos.p".format(current_ckpt_filename.split(".pth")[0]),
                    ),
                )
                self._save_info_dict(
                    current_overall_result,
                    os.path.join(self.config.INFO_DIR, "eval_infos.p"),
                )

                if self.config.EVAL.SAVE_RANKED_IMGS and self.config.VO.use_vo_model:
                    logger.info("Start post processing ...\n")
                    self._eval_ckpt_post_process(current_episode_result)
                    logger.info("... post processing done.\n")

    def _eval_ckpt_post_process(self, ckpt_eval_result):

        cur_config = ckpt_eval_result["config"]
        top_k = cur_config.EVAL.RANK_TOP_K

        for k in tqdm.tqdm(ckpt_eval_result):
            if k != "config":

                delta_type_dict = {
                    "dx": defaultdict(lambda: defaultdict(list)),
                    "dz": defaultdict(lambda: defaultdict(list)),
                    "dyaw": defaultdict(lambda: defaultdict(list)),
                }

                # sort all steps in this scene
                for episode_info in tqdm.tqdm(ckpt_eval_result[k].values()):

                    cur_map_info = episode_info["map"]

                    for tmp in episode_info["traj"]:
                        step_info = copy.deepcopy(tmp)
                        step_info["map"] = cur_map_info
                        act = ACT_IDX2NAME[step_info["action"]]
                        for i, d_type in enumerate(["dx", "dz", "dyaw"]):
                            step_info[f"{d_type}_abs"] = np.abs(
                                step_info["gt_delta"][i] - step_info["pred_delta"][i]
                            )
                            step_info[f"{d_type}_rel"] = np.abs(
                                step_info["gt_delta"][i] - step_info["pred_delta"][i]
                            ) / (np.abs(step_info["gt_delta"][i]) + EPSILON)
                            delta_type_dict[d_type][act][f"abs"].append(step_info)
                            delta_type_dict[d_type][act][f"rel"].append(step_info)

                    for d_type in ["dx", "dz", "dyaw"]:
                        for act in delta_type_dict[d_type]:
                            ranked_list_abs = delta_type_dict[d_type][act][f"abs"]
                            ranked_list_abs = sorted(
                                ranked_list_abs,
                                key=lambda x: x[f"{d_type}_abs"],
                                reverse=True,
                            )
                            delta_type_dict[d_type][act]["abs"] = ranked_list_abs[
                                :top_k
                            ]

                            ranked_list_rel = delta_type_dict[d_type][act]["rel"]
                            ranked_list_rel = sorted(
                                ranked_list_rel,
                                key=lambda x: x[f"{d_type}_rel"],
                                reverse=True,
                            )
                            delta_type_dict[d_type][act]["rel"] = ranked_list_rel[
                                :top_k
                            ]

                # plot figures
                cur_scene = os.path.basename(k).split(".")[0]
                cur_scene_dir = os.path.join(self.config.VIDEO_DIR, cur_scene)
                os.makedirs(cur_scene_dir)

                cur_config.defrost()
                cur_config.TASK_CONFIG.DATASET.CONTENT_SCENES = [cur_scene]
                cur_config.TASK_CONFIG.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
                cur_config.freeze()

                with habitat.Env(config=cur_config.TASK_CONFIG) as env:

                    for i, d_type in enumerate(["dx", "dz", "dyaw"]):
                        for compare_type in ["abs", "rel"]:
                            cur_d_dir = os.path.join(
                                cur_scene_dir, f"{d_type}_{compare_type}"
                            )
                            os.makedirs(cur_d_dir, exist_ok=False)

                            for act in delta_type_dict[d_type]:
                                ranked_list = delta_type_dict[d_type][act][compare_type]
                                assert len(ranked_list) == top_k

                                for j, step_info in enumerate(ranked_list):

                                    # obtain observation
                                    prev_obs = env._sim.get_observations_at(
                                        position=step_info["prev_agent_state"][
                                            "position"
                                        ],
                                        rotation=step_info["prev_agent_state"][
                                            "rotation"
                                        ],
                                        keep_agent_at_new_pose=False,
                                    )
                                    cur_obs = env._sim.get_observations_at(
                                        position=step_info["cur_agent_state"][
                                            "position"
                                        ],
                                        rotation=step_info["cur_agent_state"][
                                            "rotation"
                                        ],
                                        keep_agent_at_new_pose=False,
                                    )
                                    prev_rgb = prev_obs["rgb"].astype(np.uint8)
                                    cur_rgb = cur_obs["rgb"].astype(np.uint8)
                                    prev_depth = (
                                        np.repeat(prev_obs["depth"], 3, axis=2) * 255.0
                                    ).astype(np.uint8)
                                    cur_depth = (
                                        np.repeat(cur_obs["depth"], 3, axis=2) * 255.0
                                    ).astype(np.uint8)

                                    # plot map
                                    prev_top_down_map = self._get_top_down_map(
                                        step_info, "prev", cur_rgb.shape[0]
                                    )
                                    cur_top_down_map = self._get_top_down_map(
                                        step_info, "cur", cur_rgb.shape[0]
                                    )

                                    # set layout of the image
                                    first_row = np.concatenate(
                                        (prev_top_down_map, prev_rgb, prev_depth),
                                        axis=1,
                                    )
                                    second_row = np.concatenate(
                                        (cur_top_down_map, cur_rgb, cur_depth), axis=1,
                                    )
                                    out_img = np.concatenate(
                                        (first_row, second_row), axis=0,
                                    )

                                    tmp_k = f"{d_type}_{compare_type}"
                                    out_f = os.path.join(
                                        cur_d_dir,
                                        f"{act}-rank_{j:02d}-gt_{step_info['gt_delta'][i]:.3f}-"
                                        f"pred_{step_info['pred_delta'][i]:.3f}-"
                                        f"{compare_type}_{step_info[tmp_k]:.3f}-"
                                        f"collision_{step_info['collision']}.png",
                                    )
                                    imageio.imsave(out_f, out_img)

    def _get_top_down_map(self, step_info, state_k, target_size):
        map_info = step_info["map"]
        top_down_map = map_info["blank_top_down_map"]
        top_down_map = maps.colorize_topdown_map(top_down_map)

        map_agent_x, map_agent_y = maps.to_grid(
            step_info[f"{state_k}_agent_state"]["position"][0],  # x
            step_info[f"{state_k}_agent_state"]["position"][2],  # z
            map_info["coordinate_min"],
            map_info["coordinate_max"],
            map_info["map_resolution"],
        )
        agent_map_coord = (
            map_agent_x - (map_info["ind_x_min"] - map_info["grid_delta"]),
            map_agent_y - (map_info["ind_y_min"] - map_info["grid_delta"]),
        )

        if self.config.EVAL.RESIZE_TOPDOWN_MAP:
            top_down_map = resize_top_down_map(
                top_down_map,
                [[agent_map_coord, step_info[f"{state_k}_agent_angle"]]],
                target_size,
            )

        return top_down_map

    def _setup_eval_config(self, checkpoint_config: Config) -> Config:
        r"""Sets up and returns a merged config for evaluation. Config
            object saved from checkpoint is merged into config file specified
            at evaluation time with the following overwrite priority:
                  eval_opts > ckpt_opts > eval_cfg > ckpt_cfg
            If the saved config is outdated, only the eval config is returned.

        Args:
            checkpoint_config: saved config from checkpoint.

        Returns:
            Config: merged config for eval.
        """

        config = self.config.clone()

        ckpt_cmd_opts = checkpoint_config.CMD_TRAILING_OPTS
        eval_cmd_opts = config.CMD_TRAILING_OPTS

        try:
            config.merge_from_other_cfg(checkpoint_config)
            config.merge_from_other_cfg(self.config)
            config.merge_from_list(ckpt_cmd_opts)
            config.merge_from_list(eval_cmd_opts)
        except KeyError:
            logger.info("Saved config is outdated, using solely eval config")
            config = self.config.clone()
            config.merge_from_list(eval_cmd_opts)
        if config.TASK_CONFIG.DATASET.SPLIT == "train":
            config.TASK_CONFIG.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = "val"
            config.TASK_CONFIG.freeze()

        config.TASK_CONFIG.defrost()
        config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = self.config.SENSORS
        config.freeze()

        return config

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint. Trainer algorithms should
        implement this.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        test_recurrent_hidden_states,
        not_done_masks,
        current_episode_reward,
        prev_actions,
        batch,
        rgb_frames,
    ):
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[:, state_index]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                try:
                    batch[k] = v[state_index]
                except:
                    print(
                        f"\nin base_trainer.py _pause_envs(): {k}, {len(v)}, {state_index}, {envs_to_pause}\n"
                    )

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
        )

    def _save_info_dict(self, save_dict: Dict[str, List], f_path: str):
        if not os.path.isfile(f_path):
            tmp_dict = save_dict
        else:
            with open(f_path, "rb") as f:
                tmp_dict = joblib.load(f)
                for k, v in save_dict.items():
                    if k in tmp_dict:
                        tmp_dict[k].extend(v)
                    else:
                        tmp_dict[k] = v
        with open(f_path, "wb") as f:
            joblib.dump(tmp_dict, f, compress="lz4")
