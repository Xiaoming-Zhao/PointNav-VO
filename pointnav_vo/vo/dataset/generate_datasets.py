#! /usr/bin/env python

import argparse
import os
import time
import h5py
import glob
import json
import gzip
import numpy as np

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from pointnav_vo.utils.geometry_utils import (
    quaternion_to_array,
    modified_agent_state_target2ref,
)
from pointnav_vo.utils.misc_utils import ResizeCenterCropper, Resizer


LOG_INTERVAL = 50

# ALL rotations are in the [x, y, z, w] format
STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3
ACTIONS = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}


def set_up_vars(vis_size_w=256, vis_size_h=256):
    global CHUNK_SIZE, CHUNK_BYTES
    global VIS_SIZE_W, VIS_SIZE_H
    global H5PY_COMPRESS_KWARGS, H5PY_COMPRESS_KWARGS_RGB, H5PY_COMPRESS_KWARGS_DEPTH

    # CHUNK_SIZE = 1024
    CHUNK_SIZE = 256

    VIS_SIZE_W = vis_size_w
    VIS_SIZE_H = vis_size_h

    # RGB stored with uint8
    RGB_PAIR_SIZE = 2 * VIS_SIZE_W * VIS_SIZE_H * 3
    # Depth stored with float16
    DEPTH_PAIR_SIZE = 2 * VIS_SIZE_W * VIS_SIZE_H * 2
    # global CHUNK_SIZE = int(CHUNK_BYTES / (RGB_PAIR_SIZE + DEPTH_PAIR_SIZE))

    # 640MB
    # global CHUNK_BYTES = 640 * 1024 * 1024
    CHUNK_BYTES = int(np.ceil((RGB_PAIR_SIZE + DEPTH_PAIR_SIZE) * CHUNK_SIZE))

    print(f"\nEvery chunk contains {CHUNK_BYTES / (1024 * 1024)} MB data.")

    CHUNK_SHAPE0 = 1

    H5PY_COMPRESS_KWARGS = {
        # "chunks": True,
        # "compression": "lzf",
        # "compression_opts": 4
    }
    H5PY_COMPRESS_KWARGS_RGB = {
        "chunks": (CHUNK_SHAPE0, VIS_SIZE_W * VIS_SIZE_H * 3),
        "compression": "lzf",
        # "compression_opts": 4
    }
    H5PY_COMPRESS_KWARGS_DEPTH = {
        "chunks": (CHUNK_SHAPE0, VIS_SIZE_W * VIS_SIZE_H),
        "compression": "lzf",
        # "compression_opts": 4
    }


def generate_config(config):

    config.defrost()

    # add rgb sensor
    if "RGB_SENSOR" not in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.AGENT_0.SENSORS.append("RGB_SENSOR")

    # add depth sensor
    if "DEPTH_SENSOR" not in config.SIMULATOR.AGENT_0.SENSORS:
        config.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_SENSOR")

    # add more sensors
    config.TASK.SENSORS.append("HEADING_SENSOR")
    config.TASK.SENSORS.append("GPS_SENSOR")
    config.TASK.SENSORS.append("COMPASS_SENSOR")
    config.TASK.MEASUREMENTS.append("COLLISIONS")
    config.freeze()

    return config


def create_new_env(config_template, split_name, cur_scene, print_config_flag):

    config = generate_config(config_template)

    # specify the single scene
    config.defrost()
    config.DATASET.SPLIT = split_name
    config.DATASET.CONTENT_SCENES = [cur_scene]
    config.freeze()

    if print_config_flag:
        print(config)

    env = habitat.Env(config=config)

    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(env._sim, goal_radius, False)

    return env, follower


def save_data_to_disk(group, all_data):
    (
        new_episodes,
        actions,
        collisions,
        episode_start_positions,
        episode_start_rotations,
        episode_goal_positions,
        prev_rgbs,
        prev_depths,
        prev_point_goal_vecs,
        prev_episodic_gpses,
        prev_episodic_compasses,
        prev_global_positions,
        prev_global_rotations,
        cur_rgbs,
        cur_depths,
        cur_point_goal_vecs,
        cur_episodic_gpses,
        cur_episodic_compasses,
        cur_global_positions,
        cur_global_rotations,
        delta_positions,
        delta_rotations,
    ) = all_data

    group.create_dataset("new_episodes", data=new_episodes, **H5PY_COMPRESS_KWARGS)
    group.create_dataset("actions", data=actions, **H5PY_COMPRESS_KWARGS)
    group.create_dataset("collisions", data=collisions, **H5PY_COMPRESS_KWARGS)
    group.create_dataset(
        "episode_start_positions", data=episode_start_positions, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "episode_start_rotations", data=episode_start_rotations, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "episode_goal_positions", data=episode_goal_positions, **H5PY_COMPRESS_KWARGS
    )

    group.create_dataset("prev_rgbs", data=prev_rgbs, **H5PY_COMPRESS_KWARGS_RGB)
    group.create_dataset("prev_depths", data=prev_depths, **H5PY_COMPRESS_KWARGS_DEPTH)
    group.create_dataset(
        "prev_point_goal_vecs", data=prev_point_goal_vecs, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "prev_episodic_gpses", data=prev_episodic_gpses, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "prev_episodic_compasses", data=prev_episodic_compasses, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "prev_global_positions", data=prev_global_positions, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "prev_global_rotations", data=prev_global_rotations, **H5PY_COMPRESS_KWARGS
    )

    group.create_dataset("cur_rgbs", data=cur_rgbs, **H5PY_COMPRESS_KWARGS_RGB)
    group.create_dataset("cur_depths", data=cur_depths, **H5PY_COMPRESS_KWARGS_DEPTH)
    group.create_dataset(
        "cur_point_goal_vecs", data=cur_point_goal_vecs, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "cur_episodic_gpses", data=cur_episodic_gpses, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "cur_episodic_compasses", data=cur_episodic_compasses, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "cur_global_positions", data=cur_global_positions, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "cur_global_rotations", data=cur_global_rotations, **H5PY_COMPRESS_KWARGS
    )

    group.create_dataset(
        "delta_positions", data=delta_positions, **H5PY_COMPRESS_KWARGS
    )
    group.create_dataset(
        "delta_rotations", data=delta_rotations, **H5PY_COMPRESS_KWARGS
    )


def generate_one_dataset(
    config_template,
    split_name,
    scene_list,
    N,
    valid_act,
    rnd_p,
    save_f,
    obs_transformer=None,
):

    print_config_flag = True

    steps_per_scene = int(np.ceil(N / len(scene_list)))
    print(f"Each scene will provide {steps_per_scene} data points.\n")

    env = None
    new_env_flag = 0
    new_episode_flag = 0
    scene_cnt = 0
    cur_scene_steps = 0

    cnt = 0
    episode_cnt = 0
    action_cnt = {MOVE_FORWARD: 0, TURN_LEFT: 0, TURN_RIGHT: 0}
    collision_cnt = 0

    chunk_cnt = 0
    create_new_chunk = 1
    chunk_data_cnt = 0

    start_t = time.time()

    # https://portal.hdfgroup.org/display/HDF5/Chunking+in+HDF5
    # http://docs.h5py.org/en/stable/high/file.html#chunk-cache
    # https://stackoverflow.com/questions/48385256/optimal-hdf5-dataset-chunk-shape-for-reading-rows/48405220#48405220
    with h5py.File(
        save_f, "w", libver="latest", rdcc_nbytes=CHUNK_BYTES, rdcc_nslots=1e7
    ) as f:

        while cnt <= N:

            if create_new_chunk == 1:

                # reset flag and counter
                create_new_chunk = 0
                chunk_data_cnt = 0

                # save previsou chunk
                if cnt != 0:
                    group = f.create_group(f"chunk_{chunk_cnt}")
                    chunk_cnt += 1

                    print("Convert data into np.array ...")
                    tmp_start = time.time()

                    new_episodes = np.array(new_episodes, dtype=np.uint8)
                    actions = np.array(actions, dtype=np.uint8)
                    collisions = np.array(collisions, dtype=np.uint8)
                    episode_start_positions = np.array(
                        episode_start_positions, dtype=np.float16
                    )
                    episode_start_rotations = np.array(
                        episode_start_rotations, dtype=np.float16
                    )
                    episode_goal_positions = np.array(
                        episode_goal_positions, dtype=np.float16
                    )

                    prev_rgbs = np.array(prev_rgbs, dtype=np.uint8)
                    prev_depths = np.array(prev_depths, dtype=np.float16)
                    prev_point_goal_vecs = np.array(
                        prev_point_goal_vecs, dtype=np.float16
                    )
                    prev_episodic_gpses = np.array(
                        prev_episodic_gpses, dtype=np.float16
                    )
                    prev_episodic_compasses = np.array(
                        prev_episodic_compasses, dtype=np.float16
                    )
                    prev_global_positions = np.array(
                        prev_global_positions, dtype=np.float16
                    )
                    prev_global_rotations = np.array(
                        prev_global_rotations, dtype=np.float16
                    )

                    cur_rgbs = np.array(cur_rgbs, dtype=np.uint8)
                    cur_depths = np.array(cur_depths, dtype=np.float16)
                    cur_point_goal_vecs = np.array(
                        cur_point_goal_vecs, dtype=np.float16
                    )
                    cur_episodic_gpses = np.array(cur_episodic_gpses, dtype=np.float16)
                    cur_episodic_compasses = np.array(
                        cur_episodic_compasses, dtype=np.float16
                    )
                    cur_global_positions = np.array(
                        cur_global_positions, dtype=np.float16
                    )
                    cur_global_rotations = np.array(
                        cur_global_rotations, dtype=np.float16
                    )

                    delta_positions = np.array(delta_positions, dtype=np.float16)
                    delta_rotations = np.array(delta_rotations, dtype=np.float16)

                    print(f"... done ({time.time() - tmp_start:.2f}s)")

                    print("Save data to disk ...")
                    tmp_start = time.time()

                    all_data = [
                        new_episodes,
                        actions,
                        collisions,
                        episode_start_positions,
                        episode_start_rotations,
                        episode_goal_positions,
                        prev_rgbs,
                        prev_depths,
                        prev_point_goal_vecs,
                        prev_episodic_gpses,
                        prev_episodic_compasses,
                        prev_global_positions,
                        prev_global_rotations,
                        cur_rgbs,
                        cur_depths,
                        cur_point_goal_vecs,
                        cur_episodic_gpses,
                        cur_episodic_compasses,
                        cur_global_positions,
                        cur_global_rotations,
                        delta_positions,
                        delta_rotations,
                    ]
                    save_data_to_disk(group, all_data)

                    print(f"... done ({time.time() - tmp_start:.2f}s)\n")

                # create new chunk
                if N - cnt >= CHUNK_SIZE:
                    cur_chunk_size = CHUNK_SIZE
                else:
                    cur_chunk_size = N - cnt

                if cur_chunk_size <= 0:
                    break

                print(f"{chunk_cnt}th chunk size: {cur_chunk_size}\n")

                new_episodes = []
                actions = []
                collisions = []
                episode_start_positions = []
                episode_start_rotations = []
                episode_goal_positions = []

                prev_rgbs = []
                prev_depths = []
                prev_point_goal_vecs = []
                prev_episodic_gpses = []
                prev_episodic_compasses = []
                prev_global_positions = []
                prev_global_rotations = []

                cur_rgbs = []
                cur_depths = []
                cur_point_goal_vecs = []
                cur_episodic_gpses = []
                cur_episodic_compasses = []
                cur_global_positions = []
                cur_global_rotations = []

                delta_positions = []
                delta_rotations = []

            # create environment for new scene
            if env is None or cur_scene_steps >= steps_per_scene:
                if env is not None:
                    env.close()
                cur_scene = scene_list[scene_cnt]
                env, follower = create_new_env(
                    config_template, split_name, cur_scene, print_config_flag
                )
                print(f"Create environemnt for {cur_scene} successfully.\n")
                if print_config_flag:
                    print_config_flag = False

                # update some indicators
                scene_cnt += 1
                new_env_flag = 1
                cur_scene_steps = 0

            # print(f"\n{cur_scene}: {cur_scene_steps}")

            if new_env_flag == 1 or env.episode_over:
                new_env_flag = 0
                new_episode_flag = 1
                episode_cnt += 1
                prev_obs = env.reset()
                prev_agent_state = env._sim.get_agent_state()
            else:
                new_episode_flag = 0

            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )

            cur_obs = env.step(best_action)
            cur_agent_state = env._sim.get_agent_state()

            cur_metrics = env.get_metrics()

            if best_action in valid_act and np.random.binomial(1, rnd_p) == 1:

                action_cnt[best_action] += 1

                cnt += 1
                cur_scene_steps += 1

                if cur_metrics["collisions"]["is_collision"]:
                    collision_cnt += 1

                if cnt % LOG_INTERVAL == 0:
                    tmp_time_interval = time.time() - start_t
                    tmp_time_per_ep = tmp_time_interval / episode_cnt
                    tmp_remain_time = tmp_time_interval * (N - cnt) / cnt
                    print(
                        f"[{cnt} / {N}] "
                        f"ep: {episode_cnt}, {tmp_time_per_ep:.2f}s / episode; "
                        f"remain: {tmp_remain_time:.2f}s\n"
                    )

                chunk_data_cnt += 1
                if chunk_data_cnt == cur_chunk_size:
                    create_new_chunk = 1
                tmp_idx = chunk_data_cnt - 1

                # misc information
                new_episodes.append(new_episode_flag)
                actions.append(best_action)
                collisions.append(cur_metrics["collisions"]["is_collision"])
                episode_start_positions.append(
                    np.array(env.current_episode.start_position)
                )
                episode_start_rotations.append(
                    np.array(env.current_episode.start_rotation)
                )
                episode_goal_positions.append(
                    np.array(env.current_episode.goals[0].position)
                )

                prev_rgb = prev_obs["rgb"]
                prev_depth = prev_obs["depth"]
                cur_rgb = cur_obs["rgb"]
                cur_depth = cur_obs["depth"]
                if obs_transformer is not None:
                    raw_obs = np.concatenate(
                        (prev_rgb, prev_depth, cur_rgb, cur_depth,), axis=2
                    )
                    transformed_obs = obs_transformer(raw_obs).numpy()

                    prev_rgb = transformed_obs[:, :, :3]
                    prev_depth = transformed_obs[:, :, 3][:, :, np.newaxis]
                    cur_rgb = transformed_obs[:, :, 4:-1]
                    cur_depth = transformed_obs[:, :, -1][:, :, np.newaxis]

                assert prev_rgb.shape == (VIS_SIZE_H, VIS_SIZE_W, 3)
                assert prev_depth.shape == (VIS_SIZE_H, VIS_SIZE_W, 1)
                assert cur_rgb.shape == (VIS_SIZE_H, VIS_SIZE_W, 3)
                assert cur_depth.shape == (VIS_SIZE_H, VIS_SIZE_W, 1)

                # all previous information
                prev_rgbs.append(prev_rgb.reshape(-1))
                prev_depths.append(prev_depth.reshape(-1))
                prev_point_goal_vecs.append(prev_obs["pointgoal_with_gps_compass"])
                prev_episodic_gpses.append(prev_obs["gps"])
                prev_episodic_compasses.append(prev_obs["compass"])
                prev_global_positions.append(prev_agent_state.position)
                prev_global_rotations.append(
                    quaternion_to_array(prev_agent_state.rotation)
                )

                # print(cur_chunk_size, chunk_data_cnt, tmp_idx,
                #       prev_rgbs.dtype, prev_depths.dtype, prev_point_goal_vecs.dtype, prev_episodic_gpses.dtype,
                #      prev_episodic_compasses.dtype, prev_global_positions.dtype, prev_global_rotations.dtype)

                # all current information
                cur_rgbs.append(cur_rgb.reshape(-1))
                cur_depths.append(cur_depth.reshape(-1))
                cur_point_goal_vecs.append(cur_obs["pointgoal_with_gps_compass"])
                cur_episodic_gpses.append(cur_obs["gps"])
                cur_episodic_compasses.append(cur_obs["compass"])
                cur_global_positions.append(cur_agent_state.position)
                cur_global_rotations.append(
                    quaternion_to_array(cur_agent_state.rotation)
                )

                delta_state = modified_agent_state_target2ref(
                    prev_agent_state, cur_agent_state
                )
                delta_rotations.append(delta_state[0])
                delta_positions.append(delta_state[1])

            prev_obs = cur_obs
            prev_agent_state = cur_agent_state

    if env is not None:
        env.close()

    return episode_cnt, action_cnt, {"collision": collision_cnt}


def generate_datasets(
    config_template,
    act_type,
    rnd_p,
    N_dict,
    save_dir,
    train_scene_list,
    val_scene_list,
    obs_transformer=None,
):

    for name, N in N_dict.items():
        save_f = os.path.join(save_dir, f"{name}_{N_dict[name]}.h5")
        if name == "train":
            scene_list = train_scene_list
        elif name == "val":
            scene_list = val_scene_list
        else:
            raise ValueError

        if act_type == -1:
            valid_act = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]
        else:
            valid_act = [act_type]

        print(f"\nStart generating dataset for {name}...\n")
        episode_cnt, action_cnt, info_dict = generate_one_dataset(
            config_template,
            name,
            scene_list,
            N,
            valid_act,
            rnd_p,
            save_f,
            obs_transformer=obs_transformer,
        )
        print("\n...done.\n")

        print(f"Rollout {episode_cnt} episodes for {name}'s {N} data points.")
        for k, v in action_cnt.items():
            print(f"{ACTIONS[k]}: {v}")
        for k, v in info_dict.items():
            print(k, v)
        # print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_f", type=str, required=True, help="path to config file",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="directory for saving datasets",
    )
    parser.add_argument(
        "--act_type",
        type=int,
        required=True,
        default=-1,
        help="Whether specify which action to save.",
    )
    parser.add_argument(
        "--rnd_p",
        type=float,
        required=True,
        help="success probability for Bernoulli distribution",
    )
    parser.add_argument(
        "--N_list",
        nargs="+",
        type=int,
        required=True,
        help="A list of integer to specify the size of each dataset.",
    )
    parser.add_argument(
        "--name_list",
        nargs="+",
        type=str,
        required=True,
        choices=["train", "val"],
        help="A list of str to specify the name of each dataset.",
    )
    parser.add_argument(
        "--data_version", type=str, required=True, help="Version of dataset."
    )
    parser.add_argument(
        "--train_scene_dir",
        type=str,
        required=True,
        help="Paht to directory containing episode information of scenes for train.",
    )
    parser.add_argument(
        "--val_scene_dir",
        type=str,
        required=True,
        help="Paht to file containing episode information of scenes for validation.",
    )
    parser.add_argument(
        "--vis_size_w", type=int, required=True, help="Width of the observation."
    )
    parser.add_argument(
        "--vis_size_h", type=int, required=True, help="Height of the observation."
    )
    parser.add_argument(
        "--obs_transform",
        type=str,
        required=True,
        help="Which observation transformer to use.",
    )

    args = parser.parse_args()

    assert len(args.N_list) == len(args.name_list)

    set_up_vars(vis_size_w=args.vis_size_w, vis_size_h=args.vis_size_h)

    os.makedirs(args.save_dir, exist_ok=True)

    # get scene list for train, default train has content folder
    train_scene_list = list(glob.glob(os.path.join(args.train_scene_dir, "*.json.gz")))
    train_scene_list = sorted(
        [os.path.basename(_).split(".")[0] for _ in train_scene_list]
    )
    print(f"\nFind {len(train_scene_list)} scenes for training: {train_scene_list}.")

    if args.data_version == "v1":
        # get scene list for validation, default validation does not have content folder
        assert os.path.basename(args.val_scene_dir) == "val.json.gz"
        with gzip.open(args.val_scene_dir, "rt") as f:
            json_str = f.read()
        deserialized = json.loads(json_str)
        val_scene_list = []
        for tmp in deserialized["episodes"]:
            scene_name = os.path.basename(tmp["scene_id"]).split(".")[0]
            if scene_name not in val_scene_list:
                val_scene_list.append(scene_name)
        val_scene_list = sorted(val_scene_list)
    elif args.data_version == "v2":
        val_scene_list = list(glob.glob(os.path.join(args.val_scene_dir, "*.json.gz")))
        val_scene_list = sorted(
            [os.path.basename(_).split(".")[0] for _ in val_scene_list]
        )
    else:
        raise ValueError
    print(f"\nFind {len(val_scene_list)} scenes for validation: {val_scene_list}.")

    if args.obs_transform == "resize_crop":
        obs_transformer = ResizeCenterCropper(
            size=(VIS_SIZE_W, VIS_SIZE_H), channels_last=True
        )
    elif args.obs_transform == "resize":
        obs_transformer = Resizer(size=(VIS_SIZE_W, VIS_SIZE_H), channels_last=True)
    else:
        obs_transformer = None

    N_dict = dict(zip(args.name_list, args.N_list))

    config_template = config = habitat.get_config(config_paths=args.config_f)

    generate_datasets(
        config_template,
        args.act_type,
        args.rnd_p,
        N_dict,
        args.save_dir,
        train_scene_list,
        val_scene_list,
        obs_transformer=obs_transformer,
    )


if __name__ == "__main__":
    main()
