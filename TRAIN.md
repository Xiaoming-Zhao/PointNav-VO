# Training Details

## Table of Contents

- [VO Training](#visual-odometry-training)
  - [Dataset Generation](#dataset-generation)
  - [Model Training](#vo-model-training)
- [Policy Training](#navigation-policy-training)
- [Evaluate VO and Policy](#integration-of-navigation-policy-and-vo)

## Visual Odometry Training

First set the working path as
```bash
cd /path/to/this/repo
export POINTNAV_VO_ROOT=$PWD
```

### Dataset Generation

We need to generate dataset for training visual odometry model. Please make sure your disk space is enough for the generated data. With 1 million data entries, it takes about **460 GB**. 

```bash
cd ${POINTNAV_VO_ROOT}

export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
python ${POINTNAV_VO_ROOT}/pointnav_vo/vo/dataset/generate_datasets.py \
--config_f ${POINTNAV_VO_ROOT}/configs/point_nav_habitat_challenge_2020.yaml \
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content  \
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/val/content \
--save_dir ./dataset/vo_dataset \
--data_version v2 \
--vis_size_w 341 \
--vis_size_h 192 \
--obs_transform none \
--act_type -1 \
--rnd_p 1.0 \
--N_list 1000000 \
--name_list train
```

Argument explanation:

| Argument            | Usage                                                        |
| :------------------ | :----------------------------------------------------------- |
| `--config_f`        | Path to environment configuration,                           |
| `--train_scene_dir` | Path to Gibson train split from Habitat                      |
| `--val_scene_dir`   | Path to Gibson val split from Habitat                        |
| `--save_dir`        | Directory for saving generated dataset                       |
| `--data_version`    | Version of Habitat's Gibson dataset, `v2` for Habitat-Challenge 2020 |
| `--vis_size_w`      | Width of saved observation                                   |
| `--vis_size_h`      | Height of saved observation                                  |
| `--obs_transform`   | Type of observation transformer,  `none` for no transformation |
| `--act_type`        | Type of actions to be saved, `-1` for saving all actions     |
| `--rnd_p`           | Bernoulli probability, default is `1.0`, namely saving all steps |
| `--N_list`          | Sizes for train and validation dataset. Paper uses `1000000` and `50000` |
| `--name_list`       | Names for train and validation dataset, default is `train` and `val` |

### VO Model Training

We find the following training strategy is efficient, you need to modify `./configs/vo/vo_pointnav.yaml`:

- for action `move_forward`, set:
  - `VO.TRAIN.action_type = 1`
  - `VO.GEOMETRY.invariance_types = []`
- for action `turn_left` and `turn_right`:
  - 1st stage: train VO models separately for these two actions:
    - for action `move_left`: set
      - `VO.TRAIN.action_type = 2`
      - `VO.GEOMETRY.invariance_types = ["inverse_data_augment_only"]`.
    - for action `move_right`: set
      - `VO.TRAIN.action_type = 3`
      - `VO.GEOMETRY.invariance_types = ["inverse_data_augment_only"]`.
  - 2nd stage: jointly train VO models for `turn_left` and `turn_right` with geometric invariance loss, set:
    - `VO.TRAIN.action_type = [2, 3]`
    - `VO.GEOMETRY.invariance_types = ["inverse_joint_train"]`
    - `VO.MODEL.pretrained = True`
    - `VO.MODEL.pretrained_ckpt` to saved checkpoints in previous steps.


```bash
cd ${POINTNAV_VO_ROOT}

ulimit -n 65000 && \
conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type vo \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8338
```

Argument explanation:

| Argument       | Usage                                                        |
| :------------- | :----------------------------------------------------------- |
| `--repo-path`  | Specify absolute path to the root of this repo |
| `--task-type`  | Specify which task, visual odometry or navigation policy? Here is `vo` |
| `--noise`      | Specify whether data contains noises in RGB, Depth, or actuation |
| `--run-type`   | Specify whether it is `train` or `eval`                      |

##  Navigation Policy Training

```bash
cd ${POINTNAV_VO_ROOT}

export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type rl \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8338
```

Make sure `--nproc_per_node` equals `--n-gpu`. Argument explanation:

| Argument       | Usage                                                        |
| :------------- | :----------------------------------------------------------- |
| `--repo-path`  | Specify absolute path to the root of this repo |
| `--task-type`  | Specify which task, visual odometry or navigation policy? Here is `rl` |
| `--noise`      | Specify whether data contains noises in RGB, Depth, or actuation |
| `--run-type`   | Specify whether it is `train` or `eval`                      |
| `--n-gpu`      | Number of GPUs for distributed training                      |

## Integration of Navigation Policy and VO

Update `VO.REGRESS_MODEL.all_pretrained_ckpt` entry in `configs/rl/ddppo_pointnav.yaml` with path to saved checkpoints from [VO Model Training](#vo-model-training). Make sure `VO.REGRESS_MODEL`'s structure settings are aligned with specifications from `VO.MODEL` in `configs/vo/vo_pointnav.yaml`.

Use the following command to evaluate on Gibson validation split:

```bash
cd ${POINTNAV_VO_ROOT}

export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type rl \
--noise 1 \
--run-type eval \
--addr 127.0.1.1 \
--port 8338
```