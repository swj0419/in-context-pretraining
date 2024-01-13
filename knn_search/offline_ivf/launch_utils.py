import submitit
from typing import Callable, Dict
import argparse
import time
import os
import random

# TIMEOUT_MIN: int = 3 * 24 * 60 # cli arg
NUM_NODES: int = 1  # cli arg
TASKS_PER_NODE: int = 1  # cli arg
CPUS_PER_TASK: int = 80  # cli arg
GPUS_PER_NODE: int = 8  # cli arg
#OUTPUT_DIR: str = "/fsx-ust/marialomeli/jobs/"
#OUTPUT_DIR: str = "/fsx-checkpoints/marialomeli/offline_faiss/logs"# need to grab user name from env for /checkpoint/%u/jobs/
#OUTPUT_DIR: str ="/checkpoint/onellm/marialomeli/offline_faiss/logs"
OUTPUT_DIR: str = "/fsx-ralm-news/marialomeli/offline_faiss/logs"
JOB_NAME: str = "bigann_knn"


def launch_job(
    func: Callable,
    args: argparse.Namespace,
    cfg: Dict[str, str],
    n_probe: int,
    index_str: str,
) -> None:
    """
    Utils method to launch a bunch of slurm jobs to the cluster using the submitit library.
    """

    if args.run_type == "cluster":
        assert NUM_NODES >= 1
        executor = submitit.AutoExecutor(folder=OUTPUT_DIR)

        executor.update_parameters(
            nodes=NUM_NODES,
            gpus_per_node=GPUS_PER_NODE,
            cpus_per_task=CPUS_PER_TASK,
            tasks_per_node=TASKS_PER_NODE,
            name=JOB_NAME,
            slurm_partition=args.partition,
            slurm_time=70 * 60,
            # slurm_constraint=SLURM_CONSTRAINT,
        )

        job = executor.submit(func, args, cfg, n_probe, index_str)
        print(f"Job id: j{job.job_id}")
    elif args.run_type == "local":
        func(args, cfg, n_probe, index_str)

    else:
        raise ValueError(f"The run modality {args.run_type} is not currently supported.")
