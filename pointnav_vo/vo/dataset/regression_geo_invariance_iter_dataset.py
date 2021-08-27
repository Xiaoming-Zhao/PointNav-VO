#! /usr/bin/env python

import os
import h5py
import random
import time
import copy
import numpy as np
from tqdm import tqdm
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from habitat import logger
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    agent_state_target2ref,
)

from pointnav_vo.utils.geometry_utils import (
    quaternion_to_array,
    NormalizedDepth2TopDownViewHabitat,
    NormalizedDepth2TopDownViewHabitatTorch,
)
from pointnav_vo.vo.dataset.regression_iter_dataset import BaseRegressionDataset
from pointnav_vo.vo.common.common_vars import *


FloatTensor = torch.FloatTensor


class StatePairRegressionDataset(BaseRegressionDataset):
    r"""Data loader for state-pairs from Habitat simularo.
    """

    def __init__(
        self,
        eval_flag=False,
        data_file=None,
        num_workers=0,
        act_type=-1,
        vis_size_w=256,
        vis_size_h=256,
        collision="-1",
        discretize_depth="none",
        discretized_depth_channels=0,
        gen_top_down_view=False,
        top_down_view_infos={},
        geo_invariance_types=[],
        partial_data_n_splits=1,
        # data_aug=False,
    ):
        f"""Valid combination of action and geometric consistency types are:
        
        left OR right
          |-- inverse_data_augment_only
        
        left AND right
          |-- inverse_joint_train
        """

        assert (
            np.sum(
                [
                    "inverse_data_augment_only" in geo_invariance_types,
                    "inverse_joint_train" in geo_invariance_types,
                ]
            )
            <= 1
        ), f"inverse_data_augment_only and inverse_joint_train should not appear together."

        if "inverse_joint_train" in geo_invariance_types:
            assert (
                isinstance(act_type, list)
                and set(act_type) == set([TURN_LEFT, TURN_RIGHT])
            ) or (
                isinstance(act_type, int) and act_type == -1
            ), f"When enabling joint-training with geometric inversion, action types must be [left, right] OR -1."
        else:
            assert isinstance(act_type, int)
            if "inverse_data_augment_only" in geo_invariance_types:
                assert (
                    act_type != MOVE_FORWARD
                ), f"Data augmentation for geometric consistency about inversion is not suitable for forward action."

        self._eval = eval_flag
        self._data_f = data_file
        self._num_workers = num_workers
        self._act_type = act_type
        self._collision = collision
        self._geo_invariance_types = geo_invariance_types
        self._len = 0
        self._act_left_right_len = 0
        self._vis_size_w = vis_size_w
        self._vis_size_h = vis_size_h

        self._partial_data_n_splits = partial_data_n_splits

        self._gen_top_down_view = gen_top_down_view
        self._top_down_view_infos = top_down_view_infos

        # RGB stored with uint8
        self._rgb_pair_size = 2 * self._vis_size_w * self._vis_size_h * 3
        # Depth stored with float16
        self._depth_pair_size = 2 * self._vis_size_w * self._vis_size_h * 2
        with h5py.File(data_file, "r", libver="latest") as f:
            self._chunk_size = f[list(f.keys())[0]]["prev_rgbs"].shape[0]

        # for rgb + depth
        self._chunk_bytes = int(
            np.ceil((self._rgb_pair_size + self._depth_pair_size) * self._chunk_size)
        )
        # for misc information
        self._chunk_bytes += 20 * 2
        logger.info(f"\nDataset: chunk bytes {self._chunk_bytes / (1024 * 1024)} MB\n")

        self._discretize_depth = discretize_depth
        self._discretized_depth_channels = discretized_depth_channels
        if self._discretize_depth == "hard":
            self._discretized_depth_end_vals = []
            for i in np.arange(self._discretized_depth_channels):
                self._discretized_depth_end_vals.append(
                    i * 1.0 / self._discretized_depth_channels
                )
            self._discretized_depth_end_vals.append(1.0)

        logger.info("Get index mapping from h5py ...")

        all_chunk_keys = []

        with h5py.File(data_file, "r", libver="latest") as f:
            for chunk_k in tqdm(sorted(f.keys())):
                all_chunk_keys.append(chunk_k)
                valid_idxes, transform_idxes = self._get_valid_idxes(f, chunk_k)
                self._len += len(valid_idxes)

        logger.info("... done.\n")

        if not self._eval:
            random.shuffle(all_chunk_keys)
        if num_workers == 0:
            # no separate worker
            self._chunk_splits = all_chunk_keys
        elif num_workers > 0:
            self._chunk_splits = defaultdict(list)
            for i, chunk_k in enumerate(all_chunk_keys):
                self._chunk_splits[i % num_workers].append(chunk_k)
        else:
            raise ValueError

    @property
    def data_f(self):
        return self._data_f

    @property
    def num_workers(self):
        return self._num_workers

    @property
    def geo_invariance_types(self):
        return self._geo_invariance_types

    @property
    def act_left_right_len(self):
        return self._act_left_right_len

    def __len__(self):
        return int(self._len / self._partial_data_n_splits)

    def _get_valid_idxes(self, h5_f, chunk_k):
        act_left_right_idxes = np.where(
            (h5_f[chunk_k]["actions"][()] == TURN_LEFT)
            | (h5_f[chunk_k]["actions"][()] == TURN_RIGHT)
        )[0]
        self._act_left_right_len += len(act_left_right_idxes)

        transform_idxes = {}

        if isinstance(self._act_type, int):
            if self._act_type == -1:
                # all data is valid
                valid_act_idxes = np.arange(h5_f[chunk_k]["actions"].shape[0])
            elif "inverse_data_augment_only" in self._geo_invariance_types:
                # in this situation, action must be left or right
                assert self._act_type != MOVE_FORWARD
                valid_act_idxes = act_left_right_idxes
            else:
                valid_act_idxes = np.where(
                    h5_f[chunk_k]["actions"][()] == self._act_type
                )[0]
        else:
            assert isinstance(self._act_type, list)
            assert set(self._act_type) == set([TURN_LEFT, TURN_RIGHT])
            valid_act_idxes = act_left_right_idxes

        if self._collision == "-1":
            final_valid_idxes = valid_act_idxes
        else:
            raise ValueError

        return list(final_valid_idxes), transform_idxes

    def _process_data(self, chunk_i, i):

        actions = []
        rgb_pairs = []
        depth_pairs = []
        discretized_depth_pairs = []
        top_down_view_pairs = []

        delta_xs = []
        delta_ys = []
        delta_zs = []
        delta_yaws = []
        data_types = []
        dz_regress_masks = []

        chunk_idxs = []
        entry_idxs = []

        # rgb in HDF5: uint8, reshaped as a vector
        prev_rgb = self._prev_rgbs[i, :].reshape(
            (self._vis_size_h, self._vis_size_w, 3)
        )
        cur_rgb = self._cur_rgbs[i, :].reshape((self._vis_size_h, self._vis_size_w, 3))

        # depth in HDF5: float16, reshaped as a vector
        prev_depth = self._prev_depths[i, :].reshape(
            (self._vis_size_h, self._vis_size_w, 1)
        )
        cur_depth = self._cur_depths[i, :].reshape(
            (self._vis_size_h, self._vis_size_w, 1)
        )

        # discretize depth map to one-hot representation
        if self._discretize_depth == "hard":
            prev_discretized_depth = self._discretize_depth_func(i, prev_depth[..., 0])
            cur_discretized_depth = self._discretize_depth_func(i, cur_depth[..., 0])
        else:
            prev_discretized_depth = np.zeros((self._vis_size_h, self._vis_size_w, 1))
            cur_discretized_depth = np.zeros((self._vis_size_h, self._vis_size_w, 1))

        # generate top_down_view
        if self._gen_top_down_view:
            prev_top_down_view = []
            cur_top_down_view = []
            tmp_prev = torch.FloatTensor(prev_depth)
            tmp_cur = torch.FloatTensor(cur_depth)
            # opencv has issues with multiprocessing,
            # we have to move top-down-view generator inside child process to avoid permenant hang
            # https://github.com/opencv/opencv/issues/5150
            top_down_view_generator = NormalizedDepth2TopDownViewHabitat(
                **self._top_down_view_infos
            )
            prev_top_down_view.append(
                top_down_view_generator.gen_top_down_view(prev_depth)
            )
            cur_top_down_view.append(
                top_down_view_generator.gen_top_down_view(cur_depth)
            )
            prev_top_down_view = np.concatenate(prev_top_down_view, axis=2)
            cur_top_down_view = np.concatenate(cur_top_down_view, axis=2)
        else:
            prev_top_down_view = np.zeros((self._vis_size_h, self._vis_size_w, 1))
            cur_top_down_view = np.zeros((self._vis_size_h, self._vis_size_w, 1))

        # delta states of prev to cur
        delta_pos_cur_rel_to_prev = self._delta_positions[i, :]
        dx_cur_rel_to_prev = FloatTensor([delta_pos_cur_rel_to_prev[0]])
        dy_cur_rel_to_prev = FloatTensor([delta_pos_cur_rel_to_prev[1]])
        dz_cur_rel_to_prev = FloatTensor([delta_pos_cur_rel_to_prev[2]])

        delta_rotation_quaternion_cur_rel_to_prev = self._delta_rotations[i, :]
        # NOTE: must use arctan2 to get correct yaw
        dyaw_cur_rel_to_prev = FloatTensor(
            [
                2
                * np.arctan2(
                    delta_rotation_quaternion_cur_rel_to_prev[1],
                    delta_rotation_quaternion_cur_rel_to_prev[3],
                )
            ]
        )

        # three situations:
        # - no constrain on target action type
        # - this data's action type matches the target action type
        # - inverse_joint_train has been enabled
        # NOTE: be careful that this branch will NOT always execute.
        # For example,
        #   when training separate action model for TURN_LEFT and enable inverse_data_augment_only,
        #   if self._actions[i] == TURN_RIGHT, this branch will not be executed.
        if (
            (self._act_type == -1)
            or (
                isinstance(self._act_type, int) and (self._actions[i] == self._act_type)
            )
            or ("inverse_joint_train" in self._geo_invariance_types)
        ):

            # action in HDF5: uint8
            actions.append(self._actions[i])
            data_types.append(CUR_REL_TO_PREV)
            dz_regress_masks.append(1.0)

            chunk_idxs.append(chunk_i)
            entry_idxs.append(i)

            # to allow more processes, we use uint8 here to save each worker's memory usage
            rgb_pair_cur_rel_to_prev = torch.ByteTensor(
                np.concatenate([prev_rgb, cur_rgb], axis=2)
            )
            rgb_pairs.append(rgb_pair_cur_rel_to_prev)

            depth_pair_cur_rel_to_prev = FloatTensor(
                np.concatenate([prev_depth, cur_depth], axis=2)
            )
            depth_pairs.append(depth_pair_cur_rel_to_prev)

            # save memory with unin8
            discretized_depth_pair_cur_rel_to_prev = torch.ByteTensor(
                np.concatenate([prev_discretized_depth, cur_discretized_depth], axis=2)
            )
            discretized_depth_pairs.append(discretized_depth_pair_cur_rel_to_prev)

            top_down_view_pair_cur_rel_to_prev = FloatTensor(
                np.concatenate([prev_top_down_view, cur_top_down_view], axis=2)
            )
            top_down_view_pairs.append(top_down_view_pair_cur_rel_to_prev)

            delta_xs.append(dx_cur_rel_to_prev)
            delta_ys.append(dy_cur_rel_to_prev)
            delta_zs.append(dz_cur_rel_to_prev)
            delta_yaws.append(dyaw_cur_rel_to_prev)

        # valid situations:
        # - act_type != -1
        #   - inverse_data_augment_only has been enabled and self._actions[i] != self._act_type
        #   - inverse_joint_train has been enabled
        flag1 = (
            (self._act_type != -1)
            and ("inverse_data_augment_only" in self._geo_invariance_types)
            and (self._actions[i] != MOVE_FORWARD)
            and (self._actions[i] != self._act_type)
        )
        flag2 = (
            (self._act_type != -1)
            and (self._actions[i] != MOVE_FORWARD)
            and ("inverse_joint_train" in self._geo_invariance_types)
        )

        if flag1 or flag2:
            # get the opposite action from self._actions[i],
            # namely, if self._actions[i] is TURN_LEFT, add TURN_RIGHT
            tmp_act_list = [TURN_LEFT, TURN_RIGHT]
            actions.append(tmp_act_list[1 - (self._actions[i] == TURN_RIGHT)])

            chunk_idxs.append(chunk_i)
            entry_idxs.append(i)

            data_types.append(PREV_REL_TO_CUR)
            dz_regress_masks.append(1.0)

            # save memory
            rgb_pair_prev_rel_to_cur = torch.ByteTensor(
                np.concatenate([cur_rgb, prev_rgb], axis=2)
            )
            rgb_pairs.append(rgb_pair_prev_rel_to_cur)

            depth_pair_prev_rel_to_cur = FloatTensor(
                np.concatenate([cur_depth, prev_depth], axis=2)
            )
            depth_pairs.append(depth_pair_prev_rel_to_cur)

            # save memory
            discretized_depth_pair_prev_rel_to_cur = torch.ByteTensor(
                np.concatenate([cur_discretized_depth, prev_discretized_depth], axis=2)
            )
            discretized_depth_pairs.append(discretized_depth_pair_prev_rel_to_cur)

            top_down_view_pair_prev_rel_to_cur = FloatTensor(
                np.concatenate([cur_top_down_view, prev_top_down_view], axis=2)
            )
            top_down_view_pairs.append(top_down_view_pair_prev_rel_to_cur)

            prev_state = (
                quaternion_from_coeff(self._prev_global_rotations[i, :]),
                self._prev_global_positions[i, :],
            )

            cur_state = (
                self._cur_global_rotations[i, :],
                self._cur_global_positions[i, :],
            )

            delta_state_prev_rel_to_cur = agent_state_target2ref(cur_state, prev_state)
            delta_rotation_quaternion_prev_rel_to_cur = quaternion_to_array(
                delta_state_prev_rel_to_cur[0]
            )
            delta_pos_prev_rel_to_cur = delta_state_prev_rel_to_cur[1]

            dx_prev_rel_to_cur = FloatTensor([delta_pos_prev_rel_to_cur[0]])
            dy_prev_rel_to_cur = FloatTensor([delta_pos_prev_rel_to_cur[1]])
            dz_prev_rel_to_cur = FloatTensor([delta_pos_prev_rel_to_cur[2]])
            dyaw_prev_rel_to_cur = FloatTensor(
                [
                    2
                    * np.arctan2(
                        delta_rotation_quaternion_prev_rel_to_cur[1],
                        delta_rotation_quaternion_prev_rel_to_cur[3],
                    )
                ]
            )

            delta_xs.append(dx_prev_rel_to_cur)
            delta_ys.append(dy_prev_rel_to_cur)
            delta_zs.append(dz_prev_rel_to_cur)
            delta_yaws.append(dyaw_prev_rel_to_cur)

        actions = torch.Tensor(actions).long().unsqueeze(1)
        data_types = torch.Tensor(data_types).unsqueeze(1)
        rgb_pairs = torch.stack(rgb_pairs, dim=0)
        depth_pairs = torch.stack(depth_pairs, dim=0)
        discretized_depth_pairs = torch.stack(discretized_depth_pairs, dim=0)
        top_down_view_pairs = torch.stack(top_down_view_pairs, dim=0)

        delta_xs = torch.stack(delta_xs, dim=0)
        delta_ys = torch.stack(delta_ys, dim=0)
        delta_zs = torch.stack(delta_zs, dim=0)
        delta_yaws = torch.stack(delta_yaws, dim=0)
        dz_regress_masks = torch.Tensor(dz_regress_masks).unsqueeze(1)

        chunk_idxs = torch.Tensor(chunk_idxs).unsqueeze(1)
        entry_idxs = torch.Tensor(entry_idxs).unsqueeze(1)

        return (
            data_types,
            rgb_pairs,
            depth_pairs,
            discretized_depth_pairs,
            top_down_view_pairs,
            actions,
            delta_xs,
            delta_ys,
            delta_zs,
            delta_yaws,
            dz_regress_masks,
            chunk_idxs,
            entry_idxs,
        )

    def __iter__(self):
        try:
            worker_info = torch.utils.data.get_worker_info()
        except:
            worker_info = None
        if worker_info is None:
            # zero-worker data loading, return the full iterator
            chunk_list = self._chunk_splits
            worker_id = -1
        else:
            # in a worker process
            chunk_list = self._chunk_splits[worker_info.id]
            worker_id = worker_info.id

        if not self._eval:
            random.shuffle(chunk_list)

        for chunk_k in chunk_list:

            # logger.info(f"Worker {worker_id} is loading {chunk_k} into memory ...")
            # tmp_start = time.time()

            # load data into memory
            with h5py.File(
                self._data_f,
                "r",
                libver="latest",
                # rdcc_nbytes=self._chunk_bytes,
                rdcc_nslots=1e7,
            ) as f:

                # get valid indexes
                valid_idxes, _ = self._get_valid_idxes(f, chunk_k)

                self._actions = f[chunk_k]["actions"][()]
                self._prev_rgbs = f[chunk_k]["prev_rgbs"][()]
                self._cur_rgbs = f[chunk_k]["cur_rgbs"][()]
                self._prev_depths = f[chunk_k]["prev_depths"][()]
                self._cur_depths = f[chunk_k]["cur_depths"][()]
                self._delta_positions = f[chunk_k]["delta_positions"][()]
                self._delta_rotations = f[chunk_k]["delta_rotations"][()]
                # for geometric consistency
                self._prev_global_positions = f[chunk_k]["prev_global_positions"][()]
                self._prev_global_rotations = f[chunk_k]["prev_global_rotations"][()]
                self._cur_global_positions = f[chunk_k]["cur_global_positions"][()]
                self._cur_global_rotations = f[chunk_k]["cur_global_rotations"][()]

            # tmp_interval = time.time() - tmp_start
            # logger.info(f"... worker {worker_id} done ({tmp_interval:.2f}s).")

            # NOTE: must use random.shuffle instead of np.random.shuffle to make it random
            # np.random.shuffle always produces same order.
            # Similar issue here: https://discourse.allennlp.org/t/how-to-shuffle-the-data-each-batch-when-lazy-true/233
            # It seems like numpy changes shuffle behaviour:
            # - https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
            # - https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html#numpy.random.Generator.shuffle
            if not self._eval:
                random.shuffle(valid_idxes)

            for i, idx in enumerate(valid_idxes):
                if self._eval:
                    _use_cur_data = True
                else:
                    if i % self._partial_data_n_splits == 0:
                        _use_cur_data = True
                    else:
                        _use_cur_data = False

                if _use_cur_data:
                    out = self._process_data(int(chunk_k.split("_")[1]), idx)
                    yield out


def normal_collate_func(batch):

    data_types = torch.cat([i[0] for i in batch], 0)
    rgb_pairs = torch.cat([i[1] for i in batch], 0)
    depth_pairs = torch.cat([i[2] for i in batch], 0)
    discretized_depth_pairs = torch.cat([i[3] for i in batch], 0)
    top_down_view_pairs = torch.cat([i[4] for i in batch], 0)
    actions = torch.cat([i[5] for i in batch], 0)
    delta_xs = torch.cat([i[6] for i in batch], 0)
    delta_ys = torch.cat([i[7] for i in batch], 0)
    delta_zs = torch.cat([i[8] for i in batch], 0)
    delta_yaws = torch.cat([i[9] for i in batch], 0)
    dz_regress_masks = torch.cat([i[10] for i in batch], 0)

    chunk_idxs = torch.cat([i[11] for i in batch], 0)
    entry_idxs = torch.cat([i[12] for i in batch], 0)

    return (
        data_types,
        rgb_pairs,
        depth_pairs,
        discretized_depth_pairs,
        top_down_view_pairs,
        actions,
        delta_xs,
        delta_ys,
        delta_zs,
        delta_yaws,
        dz_regress_masks,
        chunk_idxs,
        entry_idxs,
    )


def fast_collate_func(data):

    batch = list(zip(*data))

    data_types = torch.cat(batch[0], dim=0)

    # rgb_pairs = [_.detach().clone() for _ in batch[1]]
    # depth_pairs = [_.detach().clone() for _ in batch[2]]
    # discretized_depth_pairs = [_.detach().clone() for _ in batch[3]]
    # top_down_view_pairs = [_.detach().clone() for _ in batch[4]]

    # NOTE: torch.cat on CPU is really slow but it is fast on GPU (https://github.com/pytorch/pytorch/issues/18634).
    # So we have an 'ugly' hack here that we return a list and postpone torch.cat until data is tranferred to GPU.
    # Please do not push tensor to GPU within collate func. Reasons:
    # 1) GPU does not have that many memory to allocate all worker's batch;
    # 2) We do not need to have all batches on GPU, we just need one at a time.
    #
    # With large probability, you may encounter 'RunTimeError: too many openfiles.'
    # This is because PyTorch default uses file_descriptor as sharing strategies
    # (https://pytorch.org/docs/master/multiprocessing.html?highlight=sharing%20strategy#sharing-strategies).
    # Each tensor needs a file descriptor in shared memory.
    # Therefore, with pushing a list of tensors into shared memory,
    # we will have #file_descriptor \approx len(rgb_pairs) * 4 * #workers (this may be the upper bound since processes are killed after completion).
    #
    # However, for Ubuntu, by default the limit number of file_descriptor is 1024.
    # You have several options to eliminate this error:
    # 1) use command `ulimit -n X` to set the limit to X. For most systems, X <= 65535
    # 2) change PyTorch's sharing stragety to file_system: torch.multiprocessing.set_sharing_strategy('file_system')
    # 3) replace all tensors with numpy.array since np.ndarray uses pickle instead of shared memory to communicate
    #    (https://github.com/pytorch/pytorch/issues/973#issuecomment-291287925)
    #    I did not benchmark this since create tensor from numpy need to copy data.
    #    I prefer to distribute all this work across workers instead of handling them in main process.
    #
    # Please note, when using file_descriptor as sharing strategy,
    # if you have many workers in total from all your programs (saying 50) and do not set ulimit properly,
    # you may encounter some weird issues, such as:
    # - training hangs forever
    # - torch.save crashes and reports file too large
    # - torch.save successfully but the checkpoint file is corrupted
    # - RuntimeError: received 0 items of ancdata
    # - RuntimeError: unable to open shared memory object
    # ...
    # I could solve them by reducing the number of workers
    # but I do not have clear explanations why they fail without proper traceback.
    #
    # Unfortunately, using file_system as sharing strategy is not perfect.
    # As PyTorch doc states that it is prone to memory leak.
    # Alough PyTorhc designs a daemon to handle this, there are still possibilities that leak will happen.
    #
    # Overall, here are your choices:
    # 1) if you care about speed, use a list and choose sharing strategy based on your own consideration
    # 2) if you do not care about speed, use torch.cat here and you are safe:
    #    rgb_pairs = torch.cat(batch[1], 0)
    #    depth_pairs = torch.cat(batch[2], 0)
    #    discretized_depth_pairs = torch.cat(batch[3], 0)
    #    top_down_view_pairs = torch.cat(batch[4], 0)

    rgb_pairs = list(batch[1])
    depth_pairs = list(batch[2])
    discretized_depth_pairs = list(batch[3])
    top_down_view_pairs = list(batch[4])

    actions = torch.cat(batch[5], dim=0)
    delta_xs = torch.cat(batch[6], dim=0)
    delta_ys = torch.cat(batch[7], dim=0)
    delta_zs = torch.cat(batch[8], dim=0)
    delta_yaws = torch.cat(batch[9], dim=0)
    dz_regress_masks = torch.cat(batch[10], dim=0)

    chunk_idxs = torch.cat(batch[11], 0)
    entry_idxs = torch.cat(batch[12], 0)

    del batch[:]

    return (
        data_types,
        rgb_pairs,
        depth_pairs,
        discretized_depth_pairs,
        top_down_view_pairs,
        actions,
        delta_xs,
        delta_ys,
        delta_zs,
        delta_yaws,
        dz_regress_masks,
        chunk_idxs,
        entry_idxs,
    )
