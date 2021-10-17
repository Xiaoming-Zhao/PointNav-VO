#! /usr/bin/env python

import os
import contextlib
import joblib
import random
import time
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict, OrderedDict

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch import autograd

import habitat
from habitat import Config, logger

from pointnav_vo.utils.tensorboard_utils import TensorboardWriter
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.vo.dataset.regression_geo_invariance_iter_dataset import (
    StatePairRegressionDataset,
    normal_collate_func,
    fast_collate_func,
)
from pointnav_vo.vo.engine.vo_cnn_engine import VOCNNBaseEngine
from pointnav_vo.vo.common.common_vars import *


TRAIN_NUM_WORKERS = 20
EVAL_NUM_WORKERS = 10
PREFETCH_FACTOR = 2
TIMEOUT = 5 * 60

DELTA_DIM = 3


@baseline_registry.register_vo_engine(name="vo_cnn_regression_geo_invariance_engine")
class VOCNNRegressionGeometricInvarianceEngine(VOCNNBaseEngine):
    @property
    def delta_types(self):
        return DEFAULT_DELTA_TYPES

    @property
    def geo_invariance_types(self):
        return self.config.VO.GEOMETRY.invariance_types

    def _set_up_model(self):

        vo_model_cls = baseline_registry.get_vo_model(self.config.VO.MODEL.name)
        assert vo_model_cls is not None, f"{self.config.VO.MODEL.name} is not supported"

        self.vo_model = {}
        for act in self._act_list:
            obs_size = (self.config.VO.VIS_SIZE_W, self.config.VO.VIS_SIZE_H)

            top_down_view_pair_channel = 0
            if "top_down_view" in self._observation_space:
                top_down_view_pair_channel = 2

            self.vo_model[act] = vo_model_cls(
                observation_space=self._observation_space,
                observation_size=obs_size,
                hidden_size=self.config.VO.MODEL.hidden_size,
                backbone=self.config.VO.MODEL.visual_backbone,
                normalize_visual_inputs=True,  # "rgb" in self._observation_space,
                output_dim=DELTA_DIM,
                dropout_p=self.config.VO.MODEL.dropout_p,
                discretized_depth_channels=self._discretized_depth_channels,
                top_down_view_pair_channel=top_down_view_pair_channel,
            )
            self.vo_model[act].to(self.device)

        if self._run_type == "train":
            if self.config.RESUME_TRAIN:
                logger.info(f"Resume training from {self.config.RESUME_STATE_FILE}")
                resume_ckpt = torch.load(self.config.RESUME_STATE_FILE)
                for act in self._act_list:
                    self.vo_model[act].load_state_dict(resume_ckpt["model_states"][act])
            elif self.config.VO.MODEL.pretrained:
                for act in self._act_list:
                    act_str = ACT_IDX2NAME[act]
                    logger.info(
                        f"Initializing {act_str} model from {self.config.VO.MODEL.pretrained_ckpt[act_str]}"
                    )
                    pretrained_ckpt = torch.load(
                        self.config.VO.MODEL.pretrained_ckpt[act_str]
                    )
                    if "model_state" in pretrained_ckpt:
                        self.vo_model[act].load_state_dict(
                            pretrained_ckpt["model_state"]
                        )
                    else:
                        self.vo_model[act].load_state_dict(
                            pretrained_ckpt["model_states"][act]
                        )
            else:
                pass

        if self._run_type == "eval":
            assert os.path.isfile(self.config.EVAL.EVAL_CKPT_PATH)
            logger.info(f"Eval {self.config.EVAL.EVAL_CKPT_PATH}")
            eval_ckpt = torch.load(self.config.EVAL.EVAL_CKPT_PATH)
            for act in self._act_list:
                self.vo_model[act].load_state_dict(eval_ckpt["model_states"][act])

        if self.verbose:
            logger.info(self.vo_model[self._act_list[0]])

        logger.info(
            "VO model's number of trainable parameters: {}".format(
                sum(
                    param.numel()
                    for param in list(self.vo_model.values())[0].parameters()
                    if param.requires_grad
                )
            )
        )

    def _set_up_optimizer(self):

        self.optimizer = {}
        for act in self._act_list:
            self.optimizer[act] = optim.Adam(
                list(
                    filter(lambda p: p.requires_grad, self.vo_model[act].parameters())
                ),
                lr=self.config.VO.TRAIN.lr,
                eps=self.config.VO.TRAIN.eps,
                weight_decay=self.config.VO.TRAIN.weight_decay,
            )

        if self.config.RESUME_TRAIN:
            resume_ckpt = torch.load(self.config.RESUME_STATE_FILE)
            for act in self._act_list:
                self.optimizer[act].load_state_dict(resume_ckpt["optim_states"][act])

    def _set_up_dataloader(self):

        self._data_collate_mode = "fast"

        if self._data_collate_mode == "fast":
            collate_func = fast_collate_func
        elif self._data_collate_mode == "normal":
            collate_func = normal_collate_func
        else:
            raise ValueError

        # https://discuss.pytorch.org/t/shuffle-issue-in-dataloader-how-to-get-the-same-data-shuffle-results-with-fixed-seed-but-different-network/45357/5
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        torch.cuda.manual_seed_all(self.config.TASK_CONFIG.SEED)

        self._train_batch_size = self.config.VO.TRAIN.batch_size

        if "discretized_depth" not in self._observation_space:
            assert self.config.VO.MODEL.discretize_depth == "none"
        if self.config.VO.MODEL.discretize_depth == "none":
            assert "discretized_depth" not in self._observation_space

        if "discretized_depth" in self._observation_space:
            self._discretized_depth_channels = (
                self.config.VO.MODEL.discretized_depth_channels
            )
        else:
            self._discretized_depth_channels = 0

        if "top_down_view" in self._observation_space:
            assert self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
            gen_top_down_view = True
            top_down_view_infos = {
                "min_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH,
                "max_depth": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH,
                "vis_size_h": self.config.VO.VIS_SIZE_H,
                "vis_size_w": self.config.VO.VIS_SIZE_W,
                "hfov_rad": self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV,
                "flag_center_crop": self.config.VO.MODEL.top_down_center_crop,
            }
        else:
            gen_top_down_view = False
            top_down_view_infos = {}

        if self.train_loader is None:
            train_dataset = StatePairRegressionDataset(
                eval_flag=False,
                data_file=self.config.VO.DATASET.TRAIN,
                num_workers=TRAIN_NUM_WORKERS,
                act_type=self._act_type,
                vis_size_w=self.config.VO.VIS_SIZE_W,
                vis_size_h=self.config.VO.VIS_SIZE_H,
                collision=self.config.VO.TRAIN.collision,
                geo_invariance_types=self.geo_invariance_types,
                discretize_depth=self.config.VO.MODEL.discretize_depth,
                discretized_depth_channels=self._discretized_depth_channels,
                gen_top_down_view=gen_top_down_view,
                top_down_view_infos=top_down_view_infos,
                partial_data_n_splits=self.config.VO.DATASET.PARTIAL_DATA_N_SPLITS,
            )

            # NOTE: must first check IterableDataset
            # since IterableDataset inherits from Dataset
            if isinstance(train_dataset, torch.utils.data.IterableDataset):
                shuffle_flag = False
            elif isinstance(train_dataset, torch.utils.data.Dataset):
                shuffle_flag = True
            else:
                raise ValueError

            self.train_loader = DataLoader(
                train_dataset,
                self._train_batch_size,
                shuffle=shuffle_flag,
                collate_fn=collate_func,
                num_workers=train_dataset.num_workers,
                drop_last=False,
                pin_memory=self._pin_memory_flag,
                timeout=TIMEOUT,
                # prefetch_factor=PREFETCH_FACTOR,
            )

        if self.eval_loader is None:
            if "EVAL_WITH_NOISE_OCC_ANT" in self.config.VO.DATASET:
                data_file_occant = self.config.VO.DATASET.EVAL_WITH_NOISE_OCC_ANT
            else:
                data_file_occant = None
            eval_dataset = StatePairRegressionDataset(
                eval_flag=True,
                data_file=self.config.VO.DATASET.EVAL,
                num_workers=EVAL_NUM_WORKERS,
                act_type=train_dataset._act_type,
                vis_size_w=train_dataset._vis_size_w,
                vis_size_h=train_dataset._vis_size_h,
                collision=train_dataset._collision,
                geo_invariance_types=train_dataset._geo_invariance_types,
                discretize_depth=train_dataset._discretize_depth,
                discretized_depth_channels=train_dataset._discretized_depth_channels,
                gen_top_down_view=train_dataset._gen_top_down_view,
                top_down_view_infos=train_dataset._top_down_view_infos,
                partial_data_n_splits=1,
            )
            self.eval_loader = DataLoader(
                eval_dataset,
                EVAL_BATCHSIZE,
                collate_fn=collate_func,
                num_workers=eval_dataset.num_workers,
                drop_last=False,
                pin_memory=self._pin_memory_flag,
            )

        if self._act_type == -1 and self.separate_eval_loaders is None:
            self.separate_eval_loaders = {}
            for act in [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]:
                # for separate evaluation, we do not add any geometric invariance
                act_dataset = StatePairRegressionDataset(
                    eval_flag=True,
                    data_file=self.config.VO.DATASET.EVAL,
                    num_workers=EVAL_NUM_WORKERS,
                    act_type=act,
                    vis_size_w=train_dataset._vis_size_w,
                    vis_size_h=train_dataset._vis_size_h,
                    collision=train_dataset._collision,
                    geo_invariance_types=[],
                    discretize_depth=train_dataset._discretize_depth,
                    discretized_depth_channels=train_dataset._discretized_depth_channels,
                    gen_top_down_view=train_dataset._gen_top_down_view,
                    top_down_view_infos=train_dataset._top_down_view_infos,
                )
                act_loader = DataLoader(
                    act_dataset,
                    EVAL_BATCHSIZE,
                    collate_fn=collate_func,
                    num_workers=act_dataset.num_workers,
                    drop_last=False,
                    pin_memory=self._pin_memory_flag,
                    timeout=TIMEOUT,
                    # prefetch_factor=PREFETCH_FACTOR,
                )
                self.separate_eval_loaders[ACT_IDX2NAME[act]] = act_loader

    def _transfer_batch(
        self,
        act_idxes,
        rgb_pairs,
        depth_pairs,
        discretized_depth_pairs,
        top_down_view_pairs,
    ):

        batch_pairs = {}

        if self._data_collate_mode == "fast":
            if "rgb" in self._observation_space:
                # rgb from dataloader has type uint8, we need to change it to FloatTensor
                batch_pairs["rgb"] = torch.cat(
                    [
                        _.float().to(self.device, non_blocking=self._pin_memory_flag)
                        for _ in rgb_pairs
                    ],
                    dim=0,
                )[act_idxes, :]
            if "depth" in self._observation_space:
                batch_pairs["depth"] = torch.cat(
                    [
                        _.to(self.device, non_blocking=self._pin_memory_flag)
                        for _ in depth_pairs
                    ],
                    dim=0,
                )[act_idxes, :]
            if "discretized_depth" in self._observation_space:
                # rgb from dataloader has type uint8, we need to change it to FloatTensor
                batch_pairs["discretized_depth"] = torch.cat(
                    [
                        _.float().to(self.device, non_blocking=self._pin_memory_flag)
                        for _ in discretized_depth_pairs
                    ],
                    dim=0,
                )[act_idxes, :]
            if "top_down_view" in self._observation_space:
                batch_pairs["top_down_view"] = torch.cat(
                    [
                        _.to(self.device, non_blocking=self._pin_memory_flag)
                        for _ in top_down_view_pairs
                    ],
                    dim=0,
                )[act_idxes, :]
        elif self._data_collate_mode == "normal":
            if "rgb" in self._observation_space:
                batch_pairs["rgb"] = (
                    rgb_pairs[act_idxes, :]
                    .float()
                    .to(self.device, non_blocking=self._pin_memory_flag)
                )
            if "depth" in self._observation_space:
                batch_pairs["depth"] = depth_pairs[act_idxes, :].to(
                    self.device, non_blocking=self._pin_memory_flag
                )
            if "discretized_depth" in self._observation_space:
                batch_pairs["discretized_depth"] = (
                    discretized_depth_pairs[act_idxes, :]
                    .float()
                    .to(self.device, non_blocking=self._pin_memory_flag)
                )
            if "top_down_view" in self._observation_space:
                batch_pairs["top_down_view"] = top_down_view_pairs[act_idxes, :].to(
                    self.device, non_blocking=self._pin_memory_flag
                )
        else:
            raise ValueError

        return batch_pairs

    def _compute_model_output(self, actions, batch_pairs, act=-1):
        # [batch, 3], [delta_x, delta_z, delta_yaw]
        if "act_embed" in self.config.VO.MODEL.name:
            actions = torch.squeeze(actions, dim=1).to(
                self.device, non_blocking=self._pin_memory_flag
            )
            pred_delta_states = self.vo_model[act](batch_pairs, actions)
        else:
            pred_delta_states = self.vo_model[act](batch_pairs)

        return pred_delta_states

    def _compute_geo_invariance_inverse_loss(self, deltas, actions, data_types):
        r"""IMPORTANT:
        This function assumes data_types has structure with alternative sequence of
        [cur_rel_to_prev_0, prev_rel_to_cur_0, cur_rel_to_prev_1, prev_rel_to_cur_1, ...]
        """

        assert (data_types[0::2] == CUR_REL_TO_PREV).all()
        assert (data_types[1::2] == PREV_REL_TO_CUR).all()

        # size: [batch, 3], order: [dx, dz, dyaw]
        actions_deduplicate = actions[
            torch.nonzero(data_types == CUR_REL_TO_PREV, as_tuple=True)[0]
        ]
        deltas_cur_rel_to_prev = deltas[
            torch.nonzero(data_types == CUR_REL_TO_PREV, as_tuple=True)[0], :
        ]
        deltas_prev_rel_to_cur = deltas[
            torch.nonzero(data_types == PREV_REL_TO_CUR, as_tuple=True)[0], :
        ]

        dyaw_prev_rel_to_cur = deltas_prev_rel_to_cur[:, 2]

        # inversion constraint for rotation: dyaw_cur_rel_to_prev = -dyaw_prev_rel_to_cur
        geo_inverse_rot_diffs = (
            deltas_cur_rel_to_prev[:, 2] + deltas_prev_rel_to_cur[:, 2]
        ) ** 2

        loss_geo_inverse_rot = torch.mean(geo_inverse_rot_diffs)
        abs_diff_geo_inverse_rot = torch.mean(
            torch.sqrt(geo_inverse_rot_diffs.detach())
        )

        # Recall the 2D rotation matrix for rotating a vector counterclockwise
        # in a right-handed coordinate system is:
        # [[ cos \theta,  -sin \theta ],
        #  [ sin \theta, cos \theta ]]
        # However, please note that Habitat uses negative-z as forward,
        # which means when we look at the 2D plane from top, it is a left-handed coordinate system.
        # There are two ways to do the rotation which gives the same result:
        #    1) negate the z value, then use the above right-handed rotation matrix
        #    2) use left-handed rotation matrix, which is used in the following
        # [batch, 2, 2]
        rot_mat_prev_rel_to_cur = torch.stack(
            (
                torch.cos(dyaw_prev_rel_to_cur),
                torch.sin(dyaw_prev_rel_to_cur),
                -1 * torch.sin(dyaw_prev_rel_to_cur),
                torch.cos(dyaw_prev_rel_to_cur),
            ),
            dim=1,
        ).reshape((-1, 2, 2))

        # inversion constraint for position: pos_prev_rel_to_cur = - R_{prev_rel_to_cur} * pos_cur_rel_to_prev
        # [batch, 2]
        pred_pos_prev_rel_to_cur = torch.matmul(
            rot_mat_prev_rel_to_cur,  # [batch, 2, 2]
            deltas_cur_rel_to_prev[:, :2].unsqueeze(-1),  # [batch, 2, 1]
        ).squeeze(-1)
        geo_inverse_pos_diffs = (
            deltas_prev_rel_to_cur[:, :2] + pred_pos_prev_rel_to_cur
        ) ** 2

        # Do not constrain dz when the action is MOVE_FORWARD
        # since the inversed value of dz in MOVE_FORWARD appears almost impossibly
        forward_idxes = torch.nonzero(
            actions_deduplicate == MOVE_FORWARD, as_tuple=True
        )[0]
        if forward_idxes.size(0) != 0:
            # NOTE: must use mask to avoid in-place operation
            # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
            mask = torch.ones(geo_inverse_pos_diffs.size()).to(
                geo_inverse_pos_diffs.device
            )
            mask[forward_idxes, 1] = 0.0
            geo_inverse_pos_diffs = mask * geo_inverse_pos_diffs
        loss_geo_inverse_pos = torch.mean(geo_inverse_pos_diffs)
        abs_diff_geo_inverse_pos = torch.mean(
            torch.sqrt(geo_inverse_pos_diffs.detach()), dim=0
        )

        loss_geo_inverse = loss_geo_inverse_rot + loss_geo_inverse_pos

        return loss_geo_inverse, abs_diff_geo_inverse_rot, abs_diff_geo_inverse_pos

    def _process_one_batch(
        self,
        batch_data,
        cur_geo_invariance_types,
        abs_diffs,
        target_magnitudes,
        relative_diffs,
        train_flag=True,
        save_pred=False,
        gt_deltas_to_save={},
        pred_deltas_to_save={},
    ):

        # rgb_pairs: [batch, size, size, 6]
        # depth_pairs: [batch, size, size, 2]
        # actions: [batch, 1]
        # delta_xs, delta_ys, delta_zs, delta_yaws: [batch, 1]
        (
            data_types,
            raw_rgb_pairs,
            raw_depth_pairs,
            raw_discretized_depth_pairs,
            raw_top_down_view_pairs,
            actions,
            delta_xs,
            delta_ys,
            delta_zs,
            delta_yaws,
            dz_regress_masks,
            chunk_idxs,
            entry_idxs,
        ) = batch_data

        if self._data_collate_mode == "fast":
            # explicitly deepcopy and reduce reference count to let dataloader processes close
            rgb_pairs = [_.clone() for _ in raw_rgb_pairs]
            depth_pairs = [_.clone() for _ in raw_depth_pairs]
            discretized_depth_pairs = [_.clone() for _ in raw_discretized_depth_pairs]
            top_down_view_pairs = [_.clone() for _ in raw_top_down_view_pairs]

            del raw_rgb_pairs[:]
            del raw_depth_pairs[:]
            del raw_discretized_depth_pairs[:]
            del raw_top_down_view_pairs[:]

            del raw_rgb_pairs
            del raw_depth_pairs
            del raw_discretized_depth_pairs
            del raw_top_down_view_pairs
        elif self._data_collate_mode == "normal":
            rgb_pairs = raw_rgb_pairs
            depth_pairs = raw_depth_pairs
            discretized_depth_pairs = raw_discretized_depth_pairs
            top_down_view_pairs = raw_top_down_view_pairs
        else:
            raise ValueError

        # fmt: off
         
        if "inverse_joint_train" in cur_geo_invariance_types:
            if train_flag:
                # sanity check, the absolute diff should all be clost to zeros
                debug_idxes = torch.nonzero(
                    (actions == TURN_LEFT) | (actions == TURN_RIGHT),
                    as_tuple=True,
                )[0]
                all_gt_deltas = torch.cat(
                    (delta_xs, delta_zs, delta_yaws), dim=1
                )
                 
                (
                    _,
                    debug_abs_diff_geo_inverse_rot,
                    debug_abs_diff_geo_inverse_pos,
                ) = self._compute_geo_invariance_inverse_loss(
                    all_gt_deltas[debug_idxes, :],
                    actions[debug_idxes, :],
                    data_types[debug_idxes, :],
                )
            else:
                debug_abs_diff_geo_inverse_rot = None
                debug_abs_diff_geo_inverse_pos = None
        else:
            debug_abs_diff_geo_inverse_rot = None
            debug_abs_diff_geo_inverse_pos = None
        # fmt: on

        if isinstance(self._act_type, int) and self._act_type != -1:
            assert torch.sum(actions - self._act_type) == 0
        if isinstance(self._act_type, list):
            assert torch.sum(actions == TURN_RIGHT) + torch.sum(
                actions == TURN_LEFT
            ) == actions.size(0)

        actions = actions.to(self.device)
        delta_xs = delta_xs.to(self.device)
        delta_zs = delta_zs.to(self.device)
        delta_yaws = delta_yaws.to(self.device)
        dz_regress_masks = dz_regress_masks.to(self.device)

        cur_batch_size = delta_xs.size(0)

        loss_weights = self._compute_loss_weights(
            actions, delta_xs, delta_zs, delta_yaws
        )

        loss = 0.0

        if "inverse_joint_train" in cur_geo_invariance_types:
            # NOTE: we need this index map because when training with geometric loss,
            # we need to first compute egomotion predictions from each action model,
            # which will not have the same order as batch data.
            # This index map will help us re-order the predictions later.
            idx_map_for_reorder = []
            all_pred_deltas = []
            all_actions = []
            all_data_types = []

        for act in self._act_list:

            if act == -1:
                act_idxes = torch.arange(actions.size(0))
            else:
                act_idxes = torch.nonzero(actions == act, as_tuple=True)[0]

            loss_weights_cur_act = {k: v[act_idxes] for k, v in loss_weights.items()}

            # [N, 3]
            batch_pairs = self._transfer_batch(
                act_idxes,
                rgb_pairs,
                depth_pairs,
                discretized_depth_pairs,
                top_down_view_pairs,
            )
            pred_delta_states = self._compute_model_output(
                actions[act_idxes, :], batch_pairs, act=act,
            )

            if "inverse_joint_train" in cur_geo_invariance_types:
                idx_map_for_reorder.extend(
                    [
                        (i, j)
                        for i, j in zip(
                            act_idxes.cpu().numpy(),
                            len(idx_map_for_reorder) + np.arange(act_idxes.size(0)),
                        )
                    ]
                )
                all_actions.append(actions[act_idxes])
                all_data_types.append(data_types[act_idxes])
                all_pred_deltas.append(pred_delta_states)

            # fmt: off
            if train_flag:
                # NOTE: the following is only applied during training
                # during eval, we sum all values together
                if len(cur_geo_invariance_types) == 0:
                    abs_diffs[act] = []
                    target_magnitudes[act] = []
                    relative_diffs[act] = []
                else:
                    abs_diffs[act] = defaultdict(list)
                    target_magnitudes[act] = defaultdict(list)
                    relative_diffs[act] = defaultdict(list)
            # fmt: on

            for i, d_type in enumerate(self.delta_types):

                if len(cur_geo_invariance_types) == 0:

                    tmp_data_types = []

                    if save_pred and i == 0:

                        # NOTE: we only need to save when i==0 since results are same

                        if act not in gt_deltas_to_save:
                            gt_deltas_to_save[act] = []
                            pred_deltas_to_save[act] = []

                        gt_deltas_to_save[act].append(
                            (
                                torch.cat(
                                    (
                                        chunk_idxs[act_idxes],
                                        entry_idxs[act_idxes],
                                        delta_xs[act_idxes, :].cpu(),
                                        delta_zs[act_idxes, :].cpu(),
                                        delta_yaws[act_idxes, :].cpu(),
                                    ),
                                    dim=1,
                                )
                                .cpu()
                                .numpy()
                            )
                        )

                        pred_deltas_to_save[act].append(
                            torch.cat(
                                (
                                    chunk_idxs[act_idxes],
                                    entry_idxs[act_idxes],
                                    pred_delta_states[act_idxes, :].detach().cpu(),
                                ),
                                dim=1,
                            ).numpy()
                        )

                    d_loss = self._compute_and_update_info(
                        act,
                        d_type,
                        i,
                        pred_delta_states,
                        [
                            delta_xs[act_idxes],
                            delta_zs[act_idxes],
                            delta_yaws[act_idxes],
                        ],
                        loss_weights_cur_act,
                        dz_regress_masks[act_idxes],
                        abs_diffs,
                        target_magnitudes,
                        relative_diffs,
                        update="append" if train_flag else "sum",
                        sum_multiplier=1 if train_flag else cur_batch_size,
                    )

                    loss += d_loss

                else:
                    tmp_data_types = [CUR_REL_TO_PREV]
                    if (
                        "inverse_data_augment_only" in cur_geo_invariance_types
                        or "inverse_joint_train" in cur_geo_invariance_types
                    ):
                        tmp_data_types.append(PREV_REL_TO_CUR)

                    for tmp_id in tmp_data_types:
                        tmp_name = DATA_TYPE_ID2STR[tmp_id]
                        tmp_idxes = torch.nonzero(
                            data_types[act_idxes] == tmp_id, as_tuple=True,
                        )[0]
                        tmp_loss_weights = {
                            k: v[tmp_idxes] for k, v in loss_weights_cur_act.items()
                        }

                        if save_pred and i == 0:
                            if act not in gt_deltas_to_save:
                                gt_deltas_to_save[act] = {}
                                pred_deltas_to_save[act] = {}
                            if tmp_id not in gt_deltas_to_save[act]:
                                gt_deltas_to_save[act][tmp_id] = []
                                pred_deltas_to_save[act][tmp_id] = []

                            # [#entries, 5]
                            gt_deltas_to_save[act][tmp_id].append(
                                torch.cat(
                                    (
                                        chunk_idxs[act_idxes][tmp_idxes],
                                        entry_idxs[act_idxes][tmp_idxes],
                                        delta_xs[act_idxes][tmp_idxes].cpu(),
                                        delta_zs[act_idxes][tmp_idxes].cpu(),
                                        delta_yaws[act_idxes][tmp_idxes].cpu(),
                                    ),
                                    dim=1,
                                ).numpy()
                            )

                            pred_deltas_to_save[act][tmp_id].append(
                                torch.cat(
                                    (
                                        chunk_idxs[act_idxes][tmp_idxes],
                                        entry_idxs[act_idxes][tmp_idxes],
                                        pred_delta_states[tmp_idxes, :].detach().cpu(),
                                    ),
                                    dim=1,
                                ).numpy()
                            )

                        d_loss = self._compute_and_update_info(
                            act,
                            d_type,
                            i,
                            pred_delta_states[tmp_idxes, :],
                            [
                                delta_xs[act_idxes][tmp_idxes],
                                delta_zs[act_idxes][tmp_idxes],
                                delta_yaws[act_idxes][tmp_idxes],
                            ],
                            tmp_loss_weights,
                            dz_regress_masks[act_idxes][tmp_idxes],
                            abs_diffs,
                            target_magnitudes,
                            relative_diffs,
                            update="append" if train_flag else "sum",
                            sum_multiplier=1 if train_flag else cur_batch_size,
                            data_type_name=tmp_name,
                        )

                        loss += d_loss

        if "inverse_joint_train" in cur_geo_invariance_types:
            all_pred_deltas = torch.cat(all_pred_deltas, dim=0)
            all_actions = torch.cat(all_actions, dim=0)
            all_data_types = torch.cat(all_data_types, dim=0)
            idx_map_for_reorder = {_[0]: _[1] for _ in idx_map_for_reorder}
            # make data aligned with original order for geometric loss computation
            all_pred_deltas = torch.stack(
                [
                    all_pred_deltas[idx_map_for_reorder[i], :]
                    for i in np.arange(all_pred_deltas.size(0))
                ],
                dim=0,
            )
            all_actions = torch.stack(
                [
                    all_actions[idx_map_for_reorder[i], :]
                    for i in np.arange(all_actions.size(0))
                ],
                dim=0,
            )
            all_data_types = torch.stack(
                [
                    all_data_types[idx_map_for_reorder[i], :]
                    for i in np.arange(all_data_types.size(0))
                ],
                dim=0,
            )
            valid_geo_inv_idxes = torch.nonzero(
                (all_actions == TURN_LEFT) | (all_actions == TURN_RIGHT), as_tuple=True,
            )[0]
            (
                loss_geo_inverse,
                abs_diff_geo_inverse_rot,
                abs_diff_geo_inverse_pos,
            ) = self._compute_geo_invariance_inverse_loss(
                all_pred_deltas[valid_geo_inv_idxes, :],
                all_actions[valid_geo_inv_idxes, :],
                all_data_types[valid_geo_inv_idxes, :],
            )
            loss += self.config.VO.GEOMETRY.loss_inv_weight * loss_geo_inverse
        else:
            abs_diff_geo_inverse_rot = None
            abs_diff_geo_inverse_pos = None

        infos_for_log = (
            abs_diffs,
            target_magnitudes,
            relative_diffs,
            abs_diff_geo_inverse_rot,
            abs_diff_geo_inverse_pos,
            debug_abs_diff_geo_inverse_rot,
            debug_abs_diff_geo_inverse_pos,
        )

        return loss, cur_batch_size, batch_pairs, tmp_data_types, infos_for_log

    def train(self):

        start_epoch = 0
        if self.config.RESUME_TRAIN:
            resume_ckpt = torch.load(self.config.RESUME_STATE_FILE)
            if "epoch" in resume_ckpt:
                start_epoch = resume_ckpt["epoch"]
            if "rnd_state" in resume_ckpt:
                random.setstate(resume_ckpt["rnd_state"])
                np.random.set_state(resume_ckpt["np_rnd_state"])
                torch.set_rng_state(resume_ckpt["torch_rnd_state"])
                torch.cuda.set_rng_state_all(resume_ckpt["torch_cuda_rnd_state"])

        nbatches = np.ceil(len(self.train_loader.dataset) / self._train_batch_size)

        grad_info_dict = OrderedDict()

        with (
            TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs)
        ) as writer:

            for epoch in tqdm(range(start_epoch, self.config.VO.TRAIN.epochs)):

                # NOTE: https://github.com/pytorch/pytorch/issues/1355#issuecomment-658660582
                train_iter = iter(self.train_loader)
                batch_i = 0

                with tqdm(total=nbatches) as pbar:

                    while True:

                        try:
                            batch_data = next(train_iter)
                        except StopIteration:
                            break

                        batch_i += 1
                        pbar.update()

                        if batch_i >= nbatches:
                            nbatches += 1
                            pbar.total = nbatches
                            pbar.refresh()

                        global_step = batch_i + epoch * nbatches

                        for act in self._act_list:
                            self.optimizer[act].zero_grad()

                        with (
                            autograd.detect_anomaly()
                            if self.config.VO.debug == 1
                            else contextlib.suppress()
                        ):

                            abs_diffs = {}
                            target_magnitudes = {}
                            relative_diffs = {}

                            (
                                loss,
                                cur_batch_size,
                                batch_pairs,
                                tmp_data_types,
                                infos_for_log,
                            ) = self._process_one_batch(
                                batch_data,
                                self.geo_invariance_types,
                                abs_diffs,
                                target_magnitudes,
                                relative_diffs,
                                train_flag=True,
                            )

                            (
                                abs_diffs,
                                target_magnitudes,
                                relative_diffs,
                                abs_diff_geo_inverse_rot,
                                abs_diff_geo_inverse_pos,
                                debug_abs_diff_geo_inverse_rot,
                                debug_abs_diff_geo_inverse_pos,
                            ) = infos_for_log

                            loss.backward()

                            if self.config.VO.TRAIN.log_grad:
                                self._log_grad(
                                    writer, global_step, grad_info_dict, d_type=d_type
                                )

                            for act in self._act_list:
                                self.optimizer[act].step()

                            self._log_lr(writer, global_step)

                            # NOTE
                            if batch_i == 10:
                                self._obs_log_func(writer, global_step, batch_pairs)

                            writer.add_scalar(
                                f"Objective/train", loss, global_step=global_step
                            )

                            if batch_i == nbatches - 1:
                                self._save_dict(
                                    {"train_objevtive": [loss.cpu().item()]},
                                    os.path.join(
                                        self.config.INFO_DIR, f"train_objective_info.p"
                                    ),
                                )

                            for act in self._act_list:
                                for i, d_type in enumerate(self.delta_types):

                                    # fmt: off
                                    log_name = f"train_{ACT_IDX2NAME[act]}"

                                    if len(self.geo_invariance_types) == 0:
                                        self._regress_log_func(
                                            writer,
                                            log_name,
                                            global_step,
                                            abs_diffs[act][i],
                                            target_magnitudes[act][i],
                                            relative_diffs[act][i],
                                            d_type=d_type,
                                        )

                                        if batch_i == nbatches - 1:
                                            self._regress_udpate_dict(
                                                log_name,
                                                abs_diffs[act][i],
                                                target_magnitudes[act][i],
                                                relative_diffs[act][i],
                                                d_type=d_type,
                                            )
                                    else:
                                        for tmp_id in tmp_data_types:
                                            tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                            self._regress_log_func(
                                                writer,
                                                f"{log_name}_{tmp_name}",
                                                global_step,
                                                abs_diffs[act][tmp_name][i],
                                                target_magnitudes[act][tmp_name][i],
                                                relative_diffs[act][tmp_name][i],
                                                d_type=d_type,
                                            )

                                            if batch_i == nbatches - 1:
                                                self._regress_udpate_dict(
                                                    f"{log_name}_{tmp_name}",
                                                    abs_diffs[act][tmp_name][i],
                                                    target_magnitudes[act][tmp_name][i],
                                                    relative_diffs[act][tmp_name][i],
                                                    d_type=d_type,
                                                )
                                    # fmt: on

                            if "inverse_joint_train" in self.geo_invariance_types:
                                self._geo_invariance_inverse_log_func(
                                    writer,
                                    "train",
                                    global_step,
                                    abs_diff_geo_inverse_rot,
                                    abs_diff_geo_inverse_pos,
                                )

                                self._geo_invariance_inverse_log_func(
                                    writer,
                                    "train_debug",
                                    global_step,
                                    debug_abs_diff_geo_inverse_rot,
                                    debug_abs_diff_geo_inverse_pos,
                                )

                                if batch_i == nbatches - 1:
                                    self._geo_invariance_inverse_udpate_dict(
                                        "train",
                                        abs_diff_geo_inverse_rot,
                                        abs_diff_geo_inverse_pos,
                                    )

                # NOTE: https://github.com/pytorch/pytorch/issues/1355#issuecomment-658660582
                del train_iter
                # prevent possible deadlock during epoch transition
                # https://github.com/open-mmlab/mmcv/blob/1cb3e36/mmcv/runner/epoch_based_runner.py#L26
                time.sleep(2)

                self.eval(
                    eval_act="no_specify",
                    epoch=epoch + 1,
                    writer=writer,
                    split_name="eval_all",
                )
                # for the sitaution where train with ALL types of actions
                if self._act_type == -1 and self.separate_eval_loaders is not None:
                    for act in self.separate_eval_loaders:
                        self.eval(
                            eval_act=act,
                            epoch=epoch + 1,
                            writer=writer,
                            split_name=f"eval_{act}",
                        )

                for act in self._act_list:
                    self.vo_model[act].train()

                self._save_ckpt(epoch + 1)

    def eval(
        self,
        eval_act="no_specify",
        epoch=0,
        writer=None,
        split_name="eval",
        save_pred=False,
        **kwargs,
    ):

        for act in self._act_list:
            self.vo_model[act].eval()

        if eval_act == "no_specify":
            eval_loader = self.eval_loader
        else:
            if eval_act in self.separate_eval_loaders:
                eval_loader = self.separate_eval_loaders[eval_act]
            else:
                raise ValueError

        eval_geo_invariance_types = eval_loader.dataset.geo_invariance_types

        total_size = 0
        total_loss = 0.0

        # fmt: off
        total_abs_diffs = {}
        total_target_magnitudes = {}
        total_relative_diffs = {}
        for act in self._act_list:
            if len(eval_geo_invariance_types) == 0:
                total_abs_diffs[act] = defaultdict(float)
                total_target_magnitudes[act] = defaultdict(float)
                total_relative_diffs[act] = defaultdict(float)
            else:
                total_abs_diffs[act] = defaultdict(lambda: defaultdict(float))
                total_target_magnitudes[act] = defaultdict(lambda: defaultdict(float))
                total_relative_diffs[act] = defaultdict(lambda: defaultdict(float))
        # fmt: on

        # for geometric consistency
        total_abs_diff_geo_inverse_rot = 0.0
        total_abs_diff_geo_inverse_pos = 0.0

        gt_deltas = {}
        pred_deltas = {}

        nbatches = np.ceil(len(eval_loader.dataset) / EVAL_BATCHSIZE)

        with tqdm(total=nbatches) as pbar:

            with torch.no_grad():

                eval_iter = iter(eval_loader)
                batch_i = 0

                while True:

                    try:
                        batch_data = next(eval_iter)
                    except StopIteration:
                        break

                    batch_i += 1
                    pbar.update()

                    if batch_i > nbatches:
                        nbatches += 1
                        pbar.total = nbatches
                        pbar.refresh()

                    (
                        loss,
                        cur_batch_size,
                        batch_pairs,
                        tmp_data_types,
                        infos_for_log,
                    ) = self._process_one_batch(
                        batch_data,
                        eval_geo_invariance_types,
                        total_abs_diffs,
                        total_target_magnitudes,
                        total_relative_diffs,
                        train_flag=False,
                        save_pred=save_pred,
                        gt_deltas_to_save=gt_deltas,
                        pred_deltas_to_save=pred_deltas,
                    )

                    total_size += cur_batch_size
                    total_loss += loss * cur_batch_size

                    (
                        abs_diffs,
                        target_magnitudes,
                        relative_diffs,
                        abs_diff_geo_inverse_rot,
                        abs_diff_geo_inverse_pos,
                        debug_abs_diff_geo_inverse_rot,
                        debug_abs_diff_geo_inverse_pos,
                    ) = infos_for_log

                    if "inverse_joint_train" in eval_geo_invariance_types:
                        total_abs_diff_geo_inverse_rot += (
                            abs_diff_geo_inverse_rot * cur_batch_size / 2
                        )
                        total_abs_diff_geo_inverse_pos += (
                            abs_diff_geo_inverse_pos * cur_batch_size / 2
                        )

                target_size = len(eval_loader.dataset)
                if self._act_type == -1 and eval_act == "no_specify":
                    if ("inverse_joint_train" in eval_geo_invariance_types) or (
                        "inverse_data_augment_only" in eval_geo_invariance_types
                    ):
                        target_size += eval_loader.dataset.act_left_right_len
                else:
                    if "inverse_joint_train" in eval_geo_invariance_types:
                        target_size += len(eval_loader.dataset)
                assert (
                    total_size == target_size
                ), f"The number of data as {total_size} does not match dataset length {target_size}."

                # NOTE
                del eval_iter
                time.sleep(2)

                if save_pred:
                    for act in gt_deltas:
                        if ("inverse_joint_train" in eval_geo_invariance_types) or (
                            "inverse_data_augment_only" in eval_geo_invariance_types
                        ):
                            for tmp_id in gt_deltas[act]:
                                gt_deltas[act][tmp_id] = np.concatenate(
                                    gt_deltas[act][tmp_id], axis=0
                                )
                                pred_deltas[act][tmp_id] = np.concatenate(
                                    pred_deltas[act][tmp_id], axis=0
                                )
                        else:
                            gt_deltas[act] = np.concatenate(gt_deltas[act], axis=0)
                            pred_deltas[act] = np.concatenate(pred_deltas[act], axis=0)
                    with open(
                        os.path.join(self.config.LOG_DIR, f"delta_gt_pred.p"), "wb"
                    ) as f:
                        joblib.dump(
                            {"gt": dict(gt_deltas), "pred": dict(pred_deltas)},
                            f,
                            compress="lz4",
                        )

                if writer is not None:
                    writer.add_scalar(
                        f"Objective/{split_name}",
                        total_loss / total_size,
                        global_step=epoch,
                    )

                    for act in self._act_list:
                        for d_type in self.delta_types:

                            # fmt: off
                            log_name = f"{split_name}_{ACT_IDX2NAME[act]}"

                            if len(eval_geo_invariance_types) == 0:
                                self._regress_log_func(
                                    writer,
                                    log_name,
                                    epoch,
                                    total_abs_diffs[act][d_type] / total_size,
                                    total_target_magnitudes[act][d_type] / total_size,
                                    total_relative_diffs[act][d_type] / total_size,
                                    d_type=d_type,
                                )
                            else:
                                for tmp_id in tmp_data_types:
                                    tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                    self._regress_log_func(
                                        writer,
                                        f"{log_name}_{tmp_name}",
                                        epoch,
                                        total_abs_diffs[act][tmp_name][d_type] / total_size,
                                        total_target_magnitudes[act][tmp_name][d_type] / total_size,
                                        total_relative_diffs[act][tmp_name][d_type] / total_size,
                                        d_type=d_type,
                                    )
                            # fmt: on

                    if "inverse_joint_train" in eval_geo_invariance_types:
                        self._geo_invariance_inverse_log_func(
                            writer,
                            split_name,
                            epoch,
                            total_abs_diff_geo_inverse_rot / (total_size / 2),
                            total_abs_diff_geo_inverse_pos / (total_size / 2),
                        )

                self._save_dict(
                    {
                        f"{split_name}_objective": [
                            (epoch, total_loss.cpu().item() / total_size)
                        ]
                    },
                    os.path.join(self.config.INFO_DIR, f"eval_objective_info.p"),
                )

                for act in self._act_list:
                    for d_type in self.delta_types:
                        # fmt: off
                        log_name = f"{split_name}_{ACT_IDX2NAME[act]}"

                        if len(eval_geo_invariance_types) == 0:
                            self._regress_udpate_dict(
                                log_name,
                                total_abs_diffs[act][d_type] / total_size,
                                total_target_magnitudes[act][d_type] / total_size,
                                total_relative_diffs[act][d_type] / total_size,
                                d_type=d_type,
                            )
                        else:
                            for tmp_id in tmp_data_types:
                                tmp_name = DATA_TYPE_ID2STR[tmp_id]
                                self._regress_udpate_dict(
                                    f"{log_name}_{tmp_name}",
                                    total_abs_diffs[act][tmp_name][d_type] / total_size,
                                    total_target_magnitudes[act][tmp_name][d_type] / total_size,
                                    total_relative_diffs[act][tmp_name][d_type] / total_size,
                                    d_type=d_type,
                                )
                        # fmt: on

                if "inverse_joint_train" in eval_geo_invariance_types:
                    self._geo_invariance_inverse_udpate_dict(
                        split_name,
                        total_abs_diff_geo_inverse_rot / (total_size / 2),
                        total_abs_diff_geo_inverse_pos / (total_size / 2),
                    )

    def _compute_and_update_info(
        self,
        act,
        d_type,
        delta_idx,
        pred_delta_states,
        gt_deltas,
        loss_weights,
        dz_regress_masks,
        abs_diffs,
        target_magnitudes,
        relative_diffs,
        update="append",
        sum_multiplier=1,
        data_type_name=None,
    ):
        (
            d_loss,
            d_abs_diffs,
            d_target_magnitudes,
            d_relative_diffs,
        ) = self._compute_loss(
            pred_delta_states[:, delta_idx].unsqueeze(1),
            gt_deltas,
            d_type=d_type,
            loss_weights=loss_weights,
            dz_regress_masks=dz_regress_masks,
        )

        if update == "append":
            if data_type_name is None:
                abs_diffs[act].append(d_abs_diffs)
                target_magnitudes[act].append(d_target_magnitudes)
                relative_diffs[act].append(d_relative_diffs)
            else:
                abs_diffs[act][data_type_name].append(d_abs_diffs)
                target_magnitudes[act][data_type_name].append(d_target_magnitudes)
                relative_diffs[act][data_type_name].append(d_relative_diffs)
        elif update == "sum":
            if data_type_name is None:
                abs_diffs[act][d_type] += d_abs_diffs * sum_multiplier
                target_magnitudes[act][d_type] += d_target_magnitudes * sum_multiplier
                relative_diffs[act][d_type] += d_relative_diffs * sum_multiplier
            else:
                abs_diffs[act][data_type_name][d_type] += d_abs_diffs * sum_multiplier
                target_magnitudes[act][data_type_name][d_type] += (
                    d_target_magnitudes * sum_multiplier
                )
                relative_diffs[act][data_type_name][d_type] += (
                    d_relative_diffs * sum_multiplier
                )

        return d_loss

    def _log_lr(self, writer, global_step):
        for tmp_i, param_group in enumerate(
            self.optimizer[self._act_list[0]].param_groups
        ):
            writer.add_scalar(
                f"LR/group_{tmp_i}", param_group["lr"], global_step=global_step,
            )

    def _geo_invariance_inverse_log_func(
        self,
        writer,
        split,
        global_step,
        abs_diff_geo_inverse_rot,
        abs_diff_geo_inverse_pos,
    ):
        writer.add_scalar(
            f"{split}_geo_invariance/inverse_abs_diff_dyaw",
            abs_diff_geo_inverse_rot,
            global_step=global_step,
        )
        writer.add_scalar(
            f"{split}_geo_invariance/inverse_abs_diff_dx",
            abs_diff_geo_inverse_pos[0],
            global_step=global_step,
        )
        writer.add_scalar(
            f"{split}_geo_invariance/inverse_abs_diff_dz",
            abs_diff_geo_inverse_pos[1],
            global_step=global_step,
        )

    def _geo_invariance_inverse_udpate_dict(
        self, split, abs_diff_geo_inverse_rot, abs_diff_geo_inverse_pos
    ):

        info_dict = defaultdict(list)

        info_dict[f"geo_invariance_inverse_abs_diff_dyaw"].append(
            abs_diff_geo_inverse_rot.item()
        )
        info_dict[f"geo_invariance_inverse_abs_diff_dx"].append(
            abs_diff_geo_inverse_pos[0].item()
        )
        info_dict[f"geo_invariance_inverse_abs_diff_dz"].append(
            abs_diff_geo_inverse_pos[1].item()
        )

        save_f = os.path.join(self.config.INFO_DIR, f"{split}_invariance_info.p")
        self._save_dict(dict(info_dict), save_f)

    def _obs_log_func(
        self, writer, global_step, batch_pairs,
    ):
        if "rgb" in self._observation_space:
            writer.add_image(
                "pre_obs/rgb",
                batch_pairs["rgb"][0, :, :, :3].cpu(),
                global_step,
                dataformats="HWC",
            )
            writer.add_image(
                "cur_obs/rgb",
                batch_pairs["rgb"][0, :, :, 3:].cpu(),
                global_step,
                dataformats="HWC",
            )
        if "depth" in self._observation_space:
            writer.add_image(
                "pre_obs/depth",
                batch_pairs["depth"][0, :, :, 0].cpu(),
                global_step,
                dataformats="HW",
            )
            writer.add_image(
                "cur_obs/depth",
                batch_pairs["depth"][0, :, :, 1].cpu(),
                global_step,
                dataformats="HW",
            )
        if "discretized_depth" in self._observation_space:
            for i in range(self.config.VO.MODEL.discretized_depth_channels):
                writer.add_image(
                    f"pre_obs/discretized_depth_{i}",
                    batch_pairs["discretized_depth"][0, :, :, i].cpu(),
                    global_step,
                    dataformats="HW",
                )
                writer.add_image(
                    f"cur_obs/discretized_depth_{i}",
                    batch_pairs["discretized_depth"][
                        0, :, :, self.config.VO.MODEL.discretized_depth_channels + i
                    ].cpu(),
                    global_step,
                    dataformats="HW",
                )
        if "top_down_view" in self._observation_space:
            writer.add_image(
                "prev_obs/top_down_view",
                # top_down_view_pairs[0, :, :, 0],
                batch_pairs["top_down_view"][0, :, :, 0].cpu(),
                global_step,
                dataformats="HW",
            )
            writer.add_image(
                "cur_obs/top_down_view",
                # top_down_view_pairs[0, :, :, 1],
                batch_pairs["top_down_view"][0, :, :, 1].cpu(),
                global_step,
                dataformats="HW",
            )

    def _save_ckpt(self, epoch):
        state = {
            "epoch": epoch,
            "config": self.config,
            "model_states": {k: self.vo_model[k].state_dict() for k in self._act_list},
            "optim_states": {k: self.optimizer[k].state_dict() for k in self._act_list},
            "rnd_state": random.getstate(),
            "np_rnd_state": np.random.get_state(),
            "torch_rnd_state": torch.get_rng_state(),
            "torch_cuda_rnd_state": torch.cuda.get_rng_state_all(),
        }

        try:
            torch.save(
                state,
                os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt_epoch_{epoch}.pth"),
            )
        except:
            import sys

            print(f"\ntotal size: {sys.getsizeof(state)}")
            for k, v in state.items():
                print(f"size of {k}: {sys.getsizeof(v)}")
