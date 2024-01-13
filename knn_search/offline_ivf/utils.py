import numpy as np
import os
from typing import Any, Dict, List
import yaml
import time
import yaml
from math import sqrt, floor
import faiss
from faiss.contrib.datasets import SyntheticDataset


def load_config(config):
    assert os.path.exists(config)
    with open(config, "r") as f:
        return yaml.safe_load(f)


def compute_recalls_at(
    approx_neighbours: np.ndarray, ground_truth: np.ndarray, n_probe: int, k_neighbours
) -> Dict[str, float]:
    """
    Recall@i: For each query, check the ground truth is among the first i nearest neighbours or not.
    """

    recall_scores = {}
    num_queries = approx_neighbours.shape[0]
    i = 1
    while i <= k_neighbours:
        dict_key = "n_probe_" + str(n_probe) + "_recall_at_" + str(i)
        recall_scores[dict_key] = (approx_neighbours[:, :i] == ground_truth[:, :1]).sum() / num_queries
        i *= 10

    return recall_scores


def open_vecs(fn):
    if not os.path.exists(fn):
        print(f"writing db vecs to {fn}...")
        return open(fn, "xb")  # if the file exists, raises FileExistsError
    return None


def close_vecs(out):
    if out is not None:
        out.close()
    return None


def faiss_sanity_check():
    ds = SyntheticDataset(256, 0, 100, 100)
    xq = ds.get_queries()
    xb = ds.get_database()
    index_cpu = faiss.IndexFlat(ds.d)
    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
    index_cpu.add(xb)
    index_gpu.add(xb)
    D_cpu, I_cpu = index_cpu.search(xq, 10)
    D_gpu, I_gpu = index_gpu.search(xq, 10)
    assert np.all(I_cpu == I_gpu), "faiss sanity check failed"
    assert np.all(np.isclose(D_cpu, D_gpu)), "faiss sanity check failed"


def margin(sample, idx_a, idx_b, D_a_b, D_a, D_b, k, k_extract, threshold):
    """
    two datasets: xa, xb; n = number of pairs
    idx_a - (np,) - query vector ids in xa
    idx_b - (np,) - query vector ids in xb
    D_a_b - (np,) - pairwise distances between xa[idx_a] and xb[idx_b]
    D_a - (np, k) - distances between vectors xa[idx_a] and corresponding nearest neighbours in xb
    D_b - (np, k) - distances between vectors xb[idx_b] and corresponding nearest neighbours in xa
    k - k nearest neighbours used for margin
    k_extract - number of nearest neighbours of each query in xb we consider for margin calculation and filtering
    threshold - margin threshold
    """

    n = sample
    nk = n * k_extract
    assert idx_a.shape == (n,)
    idx_a_k = idx_a.repeat(k_extract)
    assert idx_a_k.shape == (nk,)
    assert idx_b.shape == (nk,)
    assert D_a_b.shape == (nk,)
    assert D_a.shape == (n, k)
    assert D_b.shape == (nk, k)
    mean_a = np.mean(D_a, axis=1)
    assert mean_a.shape == (n,)
    mean_a_k = mean_a.repeat(k_extract)
    assert mean_a_k.shape == (nk,)
    mean_b = np.mean(D_b, axis=1)
    assert mean_b.shape == (nk,)
    margin = 2 * D_a_b / (mean_a_k + mean_b)
    above_threshold = margin > threshold
    print(np.count_nonzero(above_threshold))
    print(idx_a_k[above_threshold])
    print(idx_b[above_threshold])
    print(margin[above_threshold])
    return margin


def read_embeddings(fp,d,dt = np.dtype(np.float16)):

    assert os.path.exists(fp), f"file {fp} does not exist "
    fl = os.path.getsize(fp)
    nb = fl // d // dt.itemsize
    if fl == d * dt.itemsize * nb:  # no header
        return ("raw", np.memmap(fp, shape=(nb, d), dtype=dt, mode="r"))
    else:  # assume npy
        vecs = np.load(fp, mmap_mode="r")
        assert vecs.shape[1] == d
        assert vecs.dtype == dt
        return ("npy", vecs)


def add_group_args(group, *args, **kwargs):
    return group.add_argument(*args, **kwargs)


def tic(name):
    global tictoc
    tictoc = (name, time.time())
    print(name, end="\n", flush=True)


def toc():
    global tictoc
    name, t0 = tictoc
    dt = time.time() - t0
    print(f"{name}: {dt:.3f} s")
    return dt


def create_folder_and_write_results(output_path: str, filename: str, object_to_save: np.ndarray) -> None:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    full_filename = output_path + "/" + filename
    assert os.path.exists(full_filename) == False, f"File {full_filename} exists already."
    np.save(full_filename, object_to_save)


def get_intersection_cardinality_frequencies(I: np.ndarray, I_gt: np.ndarray) -> Dict[int, int]:
    """
    Computes the frequencies for the cardinalities of the intersection of neighbour indices.
    """
    nq = I.shape[0]
    res = []
    for ell in range(nq):
        res.append(len(np.intersect1d(I[ell, :], I_gt[ell, :])))
    values, counts = np.unique(res, return_counts=True)
    return dict(zip(values, counts))


def check_non_increasing_rows(D: np.ndarray) -> bool:
    nq, k = D.shape
    for i in range(nq):
        for j in range(k):
            if i >= 1:
                if D[i - 1, j] > D[i, j]:
                    return False
    return True


def find_all_divisors(n: int) -> List[int]:
    """
    Returns a list with all divisors of a number n.
    """

    divisors = []
    sqrt_n = floor(sqrt(n) + 1)
    for i in range(1, sqrt_n):
        if n % i == 0:
            if n / i == i:
                divisors.append(i)
            else:
                divisors.extend([i, n // i])
    return divisors


def is_pretransform_index(index):
    if index.__class__ == faiss.IndexPreTransform:
        assert hasattr(index, "chain")
        return True
    else:
        assert not hasattr(index, "chain")
        return False


def xbin_mmap(fname, dtype, maxn=-1):

    """
    Code from https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/dataset_io.py#L94
    mmap the competition file format for a given type of items
    """
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))

    # HACK - to handle improper header in file for private deep-1B
    # if override_d and override_d != d:
    #    print("Warning: xbin_mmap map returned d=%s, but overridig with %d" % (d, override_d))
    #    d = override_d
    # HACK

    assert os.stat(fname).st_size == 8 + n * d * np.dtype(dtype).itemsize
    if maxn > 0:
        n = min(n, maxn)
    return np.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))
