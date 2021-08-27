import os
import yacs

from habitat import Config, logger


def update_config_log(config: Config, run_type: str, log_dir: str):
    config.defrost()
    config.LOG_DIR = log_dir
    config.LOG_FILE = os.path.join(log_dir, f"{run_type}.log")
    config.INFO_DIR = os.path.join(log_dir, f"infos")
    config.CHECKPOINT_FOLDER = os.path.join(log_dir, "checkpoints")
    config.TENSORBOARD_DIR = os.path.join(log_dir, "tb")
    config.VIDEO_DIR = os.path.join(log_dir, "videos")
    config.freeze()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.INFO_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)
    os.makedirs(config.VIDEO_DIR, exist_ok=True)
    return config


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _valid_type(value, allow_cfg_node=False):
    return (type(value) in _VALID_TYPES) or (
        allow_cfg_node and isinstance(value, yacs.config.CfgNode)
    )


# CfgNodes can only contain a limited set of valid types
_VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}


def convert_cfg_to_dict(cfg_node, key_list=[]):
    if not isinstance(cfg_node, yacs.config.CfgNode):
        _assert_with_logging(
            _valid_type(cfg_node),
            "Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES
            ),
        )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_cfg_to_dict(v, key_list + [k])
        return cfg_dict
