#!/usr/bin/env python3

import os
import argparse
import random
import datetime
import glob
from tqdm import tqdm
import numpy as np

import torch

from habitat import logger

from pointnav_vo.utils.config_utils import update_config_log
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.config.rl_config.default import get_config as get_rl_config
from pointnav_vo.config.vo_config.default import get_config as get_vo_config


VIS_TYPE_DICT = {
    "rgb": "rgb",
    "depth": "d",
    "discretized_depth": "dd",
    "top_down_view": "proj",
}

GEO_SHORT_NAME = {
    "inverse_data_augment_only": "inv_aug",
    "inverse_joint_train": "inv_joint",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        choices=["rl", "vo"],
        required=True,
        help="Specify the task category of experiment.",
    )
    parser.add_argument(
        "--noise",
        type=int,
        required=True,
        help="Specify whether enable noisy environment.",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--n-gpu", type=str, required=True, help="timestamp for current executing."
    )
    parser.add_argument(
        "--cur-time", type=str, required=True, help="timestamp for current executing."
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    task_type: str,
    noise: int,
    exp_config: str,
    run_type: str,
    n_gpu: str,
    cur_time: str,
    opts=None,
) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    if task_type == "rl":
        config = get_rl_config(exp_config, opts)
        model_infos = config.RL.Policy
    elif task_type == "vo":
        config = get_vo_config(exp_config, opts)
        model_infos = config.VO.MODEL
    else:
        pass

    if task_type == "rl":
        rgb_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
        depth_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
        action_noise = "NOISE_MODEL" in config.TASK_CONFIG.SIMULATOR

    if noise == 1:
        if task_type == "vo":
            config.defrost()
            config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN_WITH_NOISE
            config.VO.DATASET.EVAL = config.VO.DATASET.EVAL_WITH_NOISE
            config.freeze()
        else:
            assert rgb_noise or depth_noise or action_noise
    elif noise == 0:
        if task_type == "vo":
            config.defrost()
            config.VO.DATASET.TRAIN = config.VO.DATASET.TRAIN_NO_NOISE
            config.VO.DATASET.EVAL = config.VO.DATASET.EVAL_NO_NOISE
            config.freeze()
        else:
            assert not rgb_noise and not depth_noise and not action_noise
    else:
        pass

    vo_pretrained_ckpt_type = "none"

    if run_type == "train":
        if config.RESUME_TRAIN:
            log_dir = os.path.join(
                os.path.dirname(config.RESUME_STATE_FILE), f"resume_{cur_time}"
            )
        else:
            # adding some tags to logging directory
            if task_type == "rl":
                log_folder_name = (
                    "{}-{}-vo_{}-noise_rgb_{}_depth_{}_act_{}-{}-model_{}-visual_{}-"
                    "rnn_{}_{}-updates_{}-minibatch_{}-ngpu_{}-proc_{}-lr_{}-{}".format(
                        task_type,
                        run_type,
                        int(config.RL.TUNE_WITH_VO),
                        int(rgb_noise),
                        int(depth_noise),
                        int(action_noise),
                        "_".join([_.strip().lower() for _ in config.SENSORS]),
                        model_infos.name,
                        model_infos.visual_backbone,
                        model_infos.rnn_backbone,
                        model_infos.num_recurrent_layers,
                        config.NUM_UPDATES,
                        config.RL.PPO.num_mini_batch,
                        n_gpu,
                        config.NUM_PROCESSES,
                        str(config.RL.PPO.lr),
                        cur_time,
                    )
                )
                if config.RL.TUNE_WITH_VO:
                    vo_pretrained_ckpt_type = config.VO.REGRESS_MODEL.pretrained_type
            elif task_type == "vo":
                if isinstance(config.VO.TRAIN.action_type, list):
                    act_str = "_".join([str(_) for _ in config.VO.TRAIN.action_type])
                else:
                    act_str = config.VO.TRAIN.action_type
                log_folder_name = (
                    "{}-noise_{}-{}-{}-dd_{}_{}-m_cen_{}-act_{}-model_{}-{}-geo_{}_inv_w_{}-"
                    "l_mult_fix_{}-{}-dpout_{}-e_{}-b_{}-lr_{}-w_de_{}-{}".format(
                        task_type,
                        noise,
                        run_type,
                        "_".join(
                            [
                                VIS_TYPE_DICT[_].strip().lower()
                                for _ in config.VO.MODEL.visual_type
                            ]
                        ),
                        config.VO.MODEL.discretize_depth,
                        config.VO.MODEL.discretized_depth_channels,
                        int(config.VO.MODEL.top_down_center_crop),
                        act_str,
                        model_infos.name,
                        model_infos.visual_backbone,
                        "_".join(
                            [
                                GEO_SHORT_NAME[str(_)]
                                for _ in config.VO.GEOMETRY.invariance_types
                            ]
                        ),
                        config.VO.GEOMETRY.loss_inv_weight,
                        int(config.VO.TRAIN.loss_weight_fixed),
                        "_".join(
                            [
                                str(_)
                                for _ in config.VO.TRAIN.loss_weight_multiplier.values()
                            ]
                        ),
                        config.VO.MODEL.dropout_p,
                        config.VO.TRAIN.epochs,
                        config.VO.TRAIN.batch_size,
                        config.VO.TRAIN.lr,
                        config.VO.TRAIN.weight_decay,
                        cur_time,
                    )
                )
            else:
                pass
            log_folder_name = f"seed_{config.TASK_CONFIG.SEED}-{log_folder_name}"
            log_dir = os.path.join(config.LOG_DIR, log_folder_name)
    elif "eval" in run_type:
        # save evaluation infos to the checkpoint's directory
        if os.path.isfile(config.EVAL.EVAL_CKPT_PATH):
            single_str = "single"
            log_dir = os.path.dirname(config.EVAL.EVAL_CKPT_PATH)
            tmp_eval_f = config.EVAL.EVAL_CKPT_PATH
        else:
            single_str = "mult"
            log_dir = config.EVAL.EVAL_CKPT_PATH
            tmp_eval_f = list(
                glob.glob(os.path.join(config.EVAL.EVAL_CKPT_PATH, "*.pth"))
            )[0]

        if task_type == "vo":
            log_dir = os.path.join(log_dir, f"eval_{cur_time}")
        elif task_type == "rl":
            tmp_config = torch.load(tmp_eval_f)["config"]
            ckpt_rgb_noise = (
                "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR
            )
            ckpt_depth_noise = (
                "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR
            )
            ckpt_action_noise = "NOISE_MODEL" in tmp_config.TASK_CONFIG.SIMULATOR
            if config.VO.USE_VO_MODEL:
                if config.VO.VO_TYPE == "REGRESS":
                    vo_pretrained_ckpt_type = config.VO.REGRESS_MODEL.pretrained_type
                else:
                    raise ValueError
            log_dir = os.path.join(
                log_dir,
                "seed_{}-{}-{}_ckpt-train_noise_rgb_{}_depth_{}_act_{}-"
                "eval_noise_rgb_{}_depth_{}_act_{}-vo_{}-mode_{}-rnd_n_{}-{}".format(
                    config.TASK_CONFIG.SEED,
                    config.EVAL.SPLIT,
                    single_str,
                    int(ckpt_rgb_noise),
                    int(ckpt_depth_noise),
                    int(ckpt_action_noise),
                    int(rgb_noise),
                    int(depth_noise),
                    int(action_noise),
                    vo_pretrained_ckpt_type,
                    config.VO.REGRESS_MODEL.mode,
                    config.VO.REGRESS_MODEL.rnd_mode_n,
                    cur_time,
                ),
            )
        else:
            pass
    else:
        raise ValueError

    if vo_pretrained_ckpt_type != "none" and config.VO.VO_TYPE == "REGRESS":
        config.defrost()
        config.VO.REGRESS_MODEL.pretrained_ckpt = config.VO.REGRESS_MODEL.all_pretrained_ckpt[
            vo_pretrained_ckpt_type
        ]
        config.freeze()

    config = update_config_log(config, run_type, log_dir)
    logger.add_filehandler(config.LOG_FILE)

    # reproducibility set up
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.cuda.manual_seed_all(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if run_type == "train":
        engine_name = config.ENGINE_NAME
    elif "eval" in run_type:
        if config.EVAL.EVAL_WITH_CKPT:
            if os.path.isfile(config.EVAL.EVAL_CKPT_PATH):
                eval_f_list = [config.EVAL.EVAL_CKPT_PATH]
            else:
                eval_f_list = list(
                    glob.glob(os.path.join(config.EVAL.EVAL_CKPT_PATH, "*.pth"))
                )
                eval_f_list = sorted(eval_f_list, key=lambda x: os.stat(x).st_mtime)
            engine_name = torch.load(eval_f_list[0])["config"].ENGINE_NAME
        else:
            raise NotImplementedError
    else:
        raise ValueError

    if task_type == "rl":
        trainer_init = baseline_registry.get_trainer(engine_name)
    elif task_type == "vo":
        trainer_init = baseline_registry.get_vo_engine(engine_name)
    else:
        trainer_init = None

    assert trainer_init is not None, f"{config.ENGINE_NAME} is not supported"

    if run_type == "train":
        trainer = trainer_init(config, run_type)
        trainer.train()
    elif "eval" in run_type:
        if task_type == "vo":
            for i, eval_f in tqdm(enumerate(eval_f_list), total=len(eval_f_list)):
                verbose = i == 0
                config.defrost()
                config.EVAL.EVAL_CKPT_PATH = eval_f
                config.freeze()
                trainer = trainer_init(config.clone(), run_type, verbose=verbose)

                if config.EVAL.EVAL_WITH_CKPT:
                    ckpt_epoch = int(
                        os.path.basename(eval_f).split("epoch_")[1].split(".")[0]
                    )
                else:
                    ckpt_epoch = 0

                for act in config.VO.EVAL.eval_acts:
                    trainer.eval(
                        eval_act=act,
                        split_name=f"eval_{act}" if act != "no_specify" else "eval",
                        epoch=ckpt_epoch,
                        save_pred=config.VO.EVAL.save_pred,
                        rank_pred=config.VO.EVAL.rank_pred,
                        rank_top_k=config.VO.EVAL.rank_top_k,
                    )
        else:
            trainer = trainer_init(config, run_type, verbose=False)
            trainer.eval()
    else:
        raise ValueError


if __name__ == "__main__":
    main()
