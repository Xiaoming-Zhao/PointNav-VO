import os
import datetime
import argparse


DEFAULT_ADDR = "127.0.1.1"
DEFAULT_PORT = "8338"

CMD_RL = "export CUDA_LAUNCH_BLOCKING=1 && \
       export PYTHONPATH={}:$PYTHONPATH && \
       python -u -m torch.distributed.launch \
       --nproc_per_node={} \
       --master_addr {} \
       --master_port {} \
       --use_env \
       {} \
       --task-type {} \
       --noise {} \
       --exp-config {} \
       --run-type {} \
       --n-gpu {} \
       --cur-time {}"

CMD_VO = "export CUDA_LAUNCH_BLOCKING=1 && \
       export PYTHONPATH={}:$PYTHONPATH && \
       python {} \
       --task-type {} \
       --noise {} \
       --exp-config {} \
       --run-type {} \
       --n-gpu {} \
       --cur-time {}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        choices=["rl", "vo"],
        required=True,
        help="Specify the category of the task",
    )
    parser.add_argument(
        "--noise",
        type=int,
        required=True,
        help="Whether adding noise into environment",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--repo-path", type=str, required=True, help="path to PointNav repo",
    )
    parser.add_argument(
        "--n_gpus", type=int, required=True, help="path to PointNav repo",
    )
    parser.add_argument(
        "--addr", type=str, required=True, help="master address",
    )
    parser.add_argument(
        "--port", type=str, required=True, help="master port",
    )

    args = parser.parse_args()

    cur_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

    if args.task_type == "rl":
        cur_config_f = os.path.join(args.repo_path, "configs/rl/ddppo_pointnav.yaml")
    elif args.task_type == "vo":
        cur_config_f = os.path.join(args.repo_path, "configs/vo/vo_pointnav.yaml")
    else:
        pass

    if "rl" in args.task_type:
        tmp_cmd = CMD_RL.format(
            args.repo_path,
            args.n_gpus,
            args.addr,
            args.port,
            # {}/point_nav/run.py
            os.path.join(args.repo_path, "pointnav_vo/run.py"),
            args.task_type,
            args.noise,
            cur_config_f,
            args.run_type,
            args.n_gpus,
            cur_time,
        )
    elif "vo" in args.task_type:
        tmp_cmd = CMD_VO.format(
            args.repo_path,
            os.path.join(args.repo_path, "pointnav_vo/run.py"),
            args.task_type,
            args.noise,
            cur_config_f,
            args.run_type,
            args.n_gpus,
            cur_time,
        )
    else:
        raise ValueError

    print("\n", tmp_cmd, "\n")

    os.system(tmp_cmd)
