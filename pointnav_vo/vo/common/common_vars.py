import numpy as np


NP_FLOAT_TYPE = "float16"
EPSILON = 1e-8
SENSOR_SIZE = 256
EVAL_BATCHSIZE = 64

N_ACTS = 4

UNIFIED = -1
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
ACT_IDX2NAME = {
    UNIFIED: "unified",
    MOVE_FORWARD: "forward",
    TURN_LEFT: "left",
    TURN_RIGHT: "right",
}
ACT_NAME2IDX = {
    "forward": MOVE_FORWARD,
    "left": TURN_LEFT,
    "right": TURN_RIGHT,
    "all": -1,
}

CUR_REL_TO_PREV = 0
PREV_REL_TO_CUR = 1
DATA_TYPE_ID2STR = {
    CUR_REL_TO_PREV: "cur_rel_to_prev",
    PREV_REL_TO_CUR: "prev_rel_to_cur",
}
DATA_TYPE_STR2ID = {
    CUR_REL_TO_PREV: "cur_rel_to_prev",
    PREV_REL_TO_CUR: "prev_rel_to_cur",
}

# [x, z, w]
NO_NOISE_DELTAS = {
    MOVE_FORWARD: [0.0, -0.25, 0.0],
    TURN_LEFT: [0.0, 0.0, np.radians(10)],
    TURN_RIGHT: [0.0, 0.0, -np.radians(10)],
}

DEFAULT_LOSS_WEIGHTS = {"dx": 1.0, "dz": 1.0, "dyaw": 1.0}

# [x, z, w]
DEFAULT_DELTA_TYPES = ["dx", "dz", "dyaw"]

EMBED_DIM = 32
RGB_PAIR_CHANNEL = 6
DEPTH_PAIR_CHANNEL = 2
TOP_DOWN_VIEW_PAIR_CHANNEL = 2
OCC_ANT_PAIR_CHANNEL = 4
DEFAULT_DELTA_STATE_SIZE = 4
