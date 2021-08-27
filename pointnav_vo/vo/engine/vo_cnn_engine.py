#! /usr/bin/env python

import os
import contextlib
import joblib
import random
import numpy as np
from tqdm import tqdm
from collections import deque, defaultdict, OrderedDict

import torch

import habitat
from habitat import Config, logger

from pointnav_vo.utils.config_utils import update_config_log
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.vo.common.common_vars import *


@baseline_registry.register_vo_engine(name="vo_cnn_base_enginer")
class VOCNNBaseEngine:
    def __init__(
        self, config: Config = None, run_type: str = "train", verbose: bool = True
    ):

        self._run_type = run_type

        self._pin_memory_flag = False

        if torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self._config = config
        self._verbose = verbose

        if run_type == "train" and self._config.RESUME_TRAIN:
            self._config = torch.load(self._config.RESUME_STATE_FILE)["config"]
            self._config.defrost()
            self._config.RESUME_TRAIN = config.RESUME_TRAIN
            self._config.RESUME_STATE_FILE = config.RESUME_STATE_FILE
            self._config.VO.TRAIN.epochs = config.VO.TRAIN.epochs

            self._config = update_config_log(self._config, run_type, config.LOG_DIR)

            self._config.freeze()

        if "eval" in run_type:
            assert os.path.isfile(self._config.EVAL.EVAL_CKPT_PATH)
            self._config = torch.load(self._config.EVAL.EVAL_CKPT_PATH)["config"]
            self._config.defrost()
            self._config.RESUME_TRAIN = False
            self._config.EVAL = config.EVAL
            self._config.VO.EVAL.save_pred = config.VO.EVAL.save_pred
            # TODO: maybe remove this update for dataset?
            self._config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN
            self._config.VO.DATASET.EVAL = config.VO.DATASET.EVAL
            self._config = update_config_log(self._config, run_type, config.LOG_DIR)
            self._config.freeze()

        self._config.defrost()

        # manual update config due to some legacy change
        if "PARTIAL_DATA_N_SPLITS" not in self._config.VO.DATASET:
            self._config.VO.DATASET.PARTIAL_DATA_N_SPLITS = 1

        self._config.freeze()

        if self.verbose:
            logger.info(f"Visual Odometry configs:\n{self._config}")

        self.flush_secs = 30

        self._observation_space = self.config.VO.MODEL.visual_type

        self._act_type = self.config.VO.TRAIN.action_type
        if isinstance(self._act_type, int):
            self._act_list = [self._act_type]
        elif isinstance(self._act_type, list):
            assert set(self._act_type) == set([TURN_LEFT, TURN_RIGHT])
            self._act_list = [TURN_LEFT, TURN_RIGHT]

        self.train_loader = None
        self.eval_loader = None
        self.separate_eval_loaders = None

        if self.verbose:
            logger.info("Setting up dataloader ...")
            self._set_up_dataloader()
            logger.info("... setting up dataloader done.\n")

            logger.info("Setting up model ...")
            self._set_up_model()
            logger.info("... setting up model done.\n")

            if self._run_type == "train":
                logger.info("Setting up optimizer ...")
                self._set_up_optimizer()
                logger.info("... setting up optimizer done.\n")
        else:
            self._set_up_dataloader()
            self._set_up_model()
            if self._run_type == "train":
                self._set_up_optimizer()

    @property
    def config(self):
        return self._config

    @property
    def verbose(self):
        return self._verbose

    def _set_up_model(self):
        raise NotImplementedError

    def _set_up_dataloader(self):
        raise NotImplementedError

    def _save_ckpt(self, epoch):
        raise NotImplementedError

    def _set_up_optimizer(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def _compute_loss(
        self,
        pred_delta_states,
        target_delta_states,
        d_type="dx",
        loss_weights=DEFAULT_LOSS_WEIGHTS,
        dz_regress_masks=None,
    ):
        delta_xs, delta_zs, delta_yaws = target_delta_states

        # NOTE: we should not use sqrt in the loss
        # since it may cause NaN in the backward
        if d_type == "dx":
            assert (
                delta_xs.size() == pred_delta_states.size()
            ), f"delta_xs: {delta_xs.size()}, pred_deltas: {pred_delta_states.size()}"
            delta_x_diffs = (delta_xs - pred_delta_states) ** 2
            loss_dx = torch.mean(delta_x_diffs * loss_weights["dx"])
            target_magnitude_dx = torch.mean(torch.abs(delta_xs)) + EPSILON
            abs_diff_dx = torch.mean(torch.sqrt(delta_x_diffs.detach()))
            relative_diff_dx = abs_diff_dx / target_magnitude_dx
            return loss_dx, abs_diff_dx, target_magnitude_dx, relative_diff_dx
        elif d_type == "dz":
            assert (
                delta_zs.size() == pred_delta_states.size()
            ), f"delta_zs: {delta_zs.size()}, pred_deltas: {pred_delta_states.size()}"
            delta_z_diffs = (delta_zs - pred_delta_states) ** 2

            if dz_regress_masks is not None:
                assert (
                    delta_zs.size() == dz_regress_masks.size()
                ), f"delta_zs: {delta_zs.size()}, dz_regress_masks: {dz_regress_masks.size()}"
                delta_z_diffs = dz_regress_masks * delta_z_diffs
                filtered_dz_idxes = torch.nonzero(
                    dz_regress_masks == 1.0, as_tuple=True
                )[0]
            else:
                filtered_dz_idxes = torch.tensor(np.arange(delta_zs.size()[0]))

            loss_dz = torch.mean(delta_z_diffs * loss_weights["dz"])
            if filtered_dz_idxes.size(0) == 0:
                target_magnitude_dz = torch.zeros(1) + EPSILON
                abs_diff_dz = torch.zeros(1)
            else:
                target_magnitude_dz = (
                    torch.mean(torch.abs(delta_zs[filtered_dz_idxes])) + EPSILON
                )
                abs_diff_dz = torch.mean(
                    torch.sqrt(delta_z_diffs.detach()[filtered_dz_idxes])
                )
            relative_diff_dz = abs_diff_dz / target_magnitude_dz
            return loss_dz, abs_diff_dz, target_magnitude_dz, relative_diff_dz
        elif d_type == "dyaw":
            assert (
                delta_yaws.size() == pred_delta_states.size()
            ), f"delta_yaws: {delta_yaws.size()}, pred_deltas: {pred_delta_states.size()}"
            delta_yaw_diffs = (delta_yaws - pred_delta_states) ** 2
            loss_dyaw = torch.mean(delta_yaw_diffs * loss_weights["dyaw"])
            target_magnitude_dyaw = torch.mean(torch.abs(delta_yaws)) + EPSILON
            abs_diff_dyaw = torch.mean(torch.sqrt(delta_yaw_diffs.detach()))
            relative_diff_dyaw = abs_diff_dyaw / target_magnitude_dyaw
            return loss_dyaw, abs_diff_dyaw, target_magnitude_dyaw, relative_diff_dyaw
        else:
            raise ValueError

    def _compute_loss_weights(self, actions, dxs, dys, dyaws):
        if (
            "loss_weight_fixed" in self.config.VO.TRAIN
            and self.config.VO.TRAIN.loss_weight_fixed
        ):
            loss_weights = {
                k: torch.ones(dxs.size()).to(dxs.device) * v
                for k, v in self.config.VO.TRAIN.loss_weight_multiplier.items()
            }
        else:
            no_noise_ds = np.array([NO_NOISE_DELTAS[int(_)] for _ in actions])
            no_noise_ds = torch.from_numpy(no_noise_ds).float().to(dxs.device)

            loss_weights = {}
            multiplier = self.config.VO.TRAIN.loss_weight_multiplier
            loss_weights["dx"] = torch.exp(
                multiplier["dx"] * torch.abs(no_noise_ds[:, 0].unsqueeze(1) - dxs)
            )
            loss_weights["dz"] = torch.exp(
                multiplier["dz"] * torch.abs(no_noise_ds[:, 1].unsqueeze(1) - dxs)
            )
            loss_weights["dyaw"] = torch.exp(
                multiplier["dyaw"] * torch.abs(no_noise_ds[:, 2].unsqueeze(1) - dxs)
            )

            for v in loss_weights.values():
                torch.all(v >= 1.0)

        return loss_weights

    def _log_grad(self, writer, global_step, grad_info_dict, d_type="dx"):

        if d_type not in grad_info_dict:
            grad_info_dict[d_type] = {}

        if isinstance(self.vo_model, dict):
            for k in self.vo_model:
                for n, p in self.vo_model[k].named_parameters():
                    if p.requires_grad:
                        writer.add_histogram(
                            f"{d_type}-Grad/{k}-{n}", p.grad.abs(), global_step
                        )
                        if f"{k}-{n}" not in grad_info_dict[d_type]:
                            grad_info_dict[d_type][f"{k}-{n}"] = []
                        grad_info_dict[d_type][f"{k}-{n}"].append(
                            p.grad.abs().mean().item()
                        )
        else:
            for n, p in self.vo_model.named_parameters():
                if p.requires_grad:
                    writer.add_histogram(
                        f"{d_type}-Grad/{n}", p.grad.abs(), global_step
                    )
                    if n not in grad_info_dict[d_type]:
                        grad_info_dict[d_type][n] = []
                    grad_info_dict[d_type][n].append(p.grad.abs().mean().item())

        if global_step > 0 and global_step % self.config.LOG_INTERVAL == 0:
            self._save_dict(
                grad_info_dict[d_type],
                os.path.join(self.config.LOG_DIR, f"avg_abs_grad_model_{d_type}.p"),
            )
            grad_info_dict[d_type] = {}

    def _regress_log_func(
        self,
        writer,
        split,
        global_step,
        abs_diff,
        target_magnitude,
        relative_diff,
        d_type="dx",
    ):

        writer.add_scalar(
            f"{split}_regression/{d_type}_abs_diff", abs_diff, global_step=global_step
        )
        writer.add_scalar(
            f"{split}_regression/{d_type}_target_magnitude",
            target_magnitude,
            global_step=global_step,
        )
        writer.add_scalar(
            f"{split}_regression/{d_type}_relative_diff",
            relative_diff,
            global_step=global_step,
        )

    def _regress_udpate_dict(
        self, split, abs_diffs, target_magnitudes, relative_diffs, d_type="dx"
    ):

        info_dict = defaultdict(list)

        info_dict[f"abs_diff_{d_type}"].append(abs_diffs.item())
        info_dict[f"target_{d_type}_magnitude"].append(target_magnitudes.item())
        info_dict[f"relative_diff_{d_type}"].append(relative_diffs.item())

        save_f = os.path.join(self.config.INFO_DIR, f"{split}_regression_info.p")
        self._save_dict(dict(info_dict), save_f)

    def _save_dict(self, save_dict, f_path):
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
