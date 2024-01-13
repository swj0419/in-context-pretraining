import numpy as np
from typing import List, Dict, Optional, Tuple
from utils import load_config
from dataset import create_dataset_from_oivf_config
import os
import faiss

FP16_BYTE_SIZE: int = 2


class NeighbourVectorFetcher(object):
    """
    Class responsible of finding the file where the nearest neighbour vector is located and loading the corresponding idx-th neigbour's vector.
    The vectors are stored in separate files and each file contains x number of vectors, where
    x can be read from the yaml config.
    """

    def __init__(
        self,
        queries_neighbour_indices_path: str,
        config: str,
        xb: str,
        xq: Optional[str] = None,
    ):
        self.cfg = load_config(config)
        self.xb = xb
        if xq:
            self.xq = xq
        else:
            self.xq = xb

        self.queries_neighbour_indices_path = queries_neighbour_indices_path
        self.metric = self.cfg["metric"]

    def compute_neighbours_distances(self):
        """
        Computes the nearest neighbours distances for all queries.
        """

        I = np.load(self.queries_neighbour_indices_path, allow_pickle=True)
        nq = I.shape[0]
        k = I.shape[1]

        xq_subset = create_dataset_from_oivf_config(self.cfg, self.xq).sample()
        assert xq_subset.shape[0] == I.shape[0], "check the number of queries used."

        xb_ds = create_dataset_from_oivf_config(self.cfg, self.xb)

        all_dist = np.zeros((1, k))
        for i in range(nq):
            dist = []
            neigh_idx = I[i, :]
            db_vectors = xb_ds.get(neigh_idx)
            for db_vector in db_vectors:
                if self.metric == faiss.METRIC_INNER_PRODUCT:
                    dist.append(np.inner(xq_subset[i], db_vector))
                elif self.metric == faiss.METRIC_L2:
                    dist.append(np.linalg.norm(xq_subset[i] - db_vector, ord=2) ** 2)
                else:
                    raise ValueError(f"metric not supported {self.metric}")
            all_dist = np.vstack((all_dist, dist))  # num_queriesxk
        return all_dist[1:, :]
