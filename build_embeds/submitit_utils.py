from omegaconf import DictConfig
import functools
import submitit
import logging
import time
import os

from retro_z_utils import init_logging


class WorkerFunctor(submitit.helpers.Checkpointable):
    def __init__(self, fn, *args, **kwargs):
        self.func = functools.partial(fn, *args, **kwargs)

    def __call__(self):
        init_logging()
        return self.func()


def create_executor(submitit_cfg: DictConfig):
    executor = submitit.AutoExecutor(folder=submitit_cfg.submitit_path, cluster=submitit_cfg.cluster)

    executor.update_parameters(cpus_per_task=submitit_cfg.cpus_per_task, gpus_per_node=submitit_cfg.gpus_per_node,
                               slurm_partition=submitit_cfg.partition, slurm_time=submitit_cfg.slurm_time,
                               slurm_job_name=submitit_cfg.slurm_job_name)
    # slurm_mem=submitit_cfg.slurm_mem, slurm_constraint=submitit_cfg.slurm_constraint

    excludes = os.environ.get('SLURM_EXCLUDE', None)
    if excludes is not None:
        logging.info('Excluding nodes: %s', excludes)
        executor.update_parameters(slurm_exclude=excludes)

    return executor


def await_completion_of_jobs(jobs):
    since_last_force = 0
    # FIXME:
    # with Progress() as progress:
    # FIXME:
    # task = progress.add_task('Slurm Jobs', total=len(jobs))
    while True:
        time.sleep(10)
        if since_last_force > 12:
            force = True
            since_last_force = 0
        else:
            force = False
            since_last_force += 1
        if force:
            jobs[0].done(force_check=True)
        num_complete = sum(job.done() for job in jobs)
        # FIXME:
        # progress.update(task, completed=num_complete)
        if num_complete == len(jobs):
            break


def fetch_and_validate_embedding_results(jobs):
    # job.results() is broken
    outputs = [j.result() for j in jobs]
    logging.info(f'Fetching outputs for {len(outputs)} jobs')
    num_chunks = set()
    processed_chunks = 0
    for shard_num_chunks, shard_processed_chunks in outputs:
        num_chunks.add(shard_num_chunks)
        processed_chunks += shard_processed_chunks

    if len(num_chunks) != 1:
        raise ValueError(f'num_chunks not the same for all workers: {num_chunks}')

    num_chunks = num_chunks.pop()
    if num_chunks != processed_chunks:
        raise ValueError(f'num_chunks != processed_chunks: {num_chunks} != {processed_chunks}')

    return processed_chunks


def fetch_and_validate_neighbors_results(jobs):
    # job.results() is broken
    outputs = []
    for job in jobs:
        try:
            node_results = job.result()
            outputs.extend(node_results)
        except Exception as ex:
            logging.critical(f'Error in job {job}: {ex}')
            breakpoint()  # option to try to salvage partial results, e.g. outputs.extend([None] * 10)
            raise ex
            pass  # option to jump past exception
    logging.info(f'Fetched outputs for {len(outputs)} jobs:\n{[job.job_id for job in jobs]}\noutputs:\n{outputs}')

    return outputs
