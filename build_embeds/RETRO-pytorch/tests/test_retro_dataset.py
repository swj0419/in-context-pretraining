from unittest.mock import patch

import numpy as np
import torch

from retro_pytorch.data import RETRODataset


def test_dropping_end_token_in_chunks():
    EOS_ID = 10
    chunks = np.asarray([[1, 2, 3], [3, EOS_ID, 5], [5, 6, EOS_ID], [EOS_ID, 0, 0]], dtype=np.int32)

    knn = np.asarray([[1, 2], [2, -1], [-1, -1], [-1, -1]], dtype=np.int32)

    seqs = np.asarray([0, 2], dtype=np.int32)

    chunk_size = chunks.shape[-1] - 1
    num_chunk_per_seq = 2
    seq_len = chunk_size * num_chunk_per_seq
    num_chunks = chunks.shape[0]
    num_seqs = num_chunks // num_chunk_per_seq
    assert num_seqs == seqs.shape[0]

    chunks_memmap_path = "path/to/dummy/chunks"
    knn_memmap_path = "path/to/dummy/knn"
    seqs_memmap_path = "path/to/seq"

    memmap_ndarrays = {chunks_memmap_path: chunks, knn_memmap_path: knn, seqs_memmap_path: seqs}

    def memmap_provider(path, dtype, shape, mode):
        return memmap_ndarrays[path]

    with patch("retro_pytorch.utils.np.memmap", side_effect=memmap_provider) as data_memmap:
        dataset = RETRODataset(
            total_num_sequences=num_seqs,
            num_sequences=num_seqs,
            sequences_offset=0,
            num_chunks=num_chunks,
            num_neighbors=knn,
            chunk_size=chunk_size,
            seq_len=seq_len,
            chunk_memmap_path=chunks_memmap_path,
            chunk_nn_memmap_path=knn_memmap_path,
            seq_memmap_path=seqs_memmap_path,
            retrieve=True,
            eos_id=EOS_ID,
        )

        assert len(dataset) == num_seqs

        actual_seq_tokens, actual_retrieved_tokens = dataset[0]

        assert actual_retrieved_tokens.dtype == torch.int64
        assert actual_seq_tokens.dtype == torch.int64

        expected_seq_tokens = [1, 2, 3, EOS_ID, 0]  # padding tokens after EOS token
        expected_retrieved_tokens = [
            [
                [3, EOS_ID, 0, 0],  # the content of chunk[1:3] padding after EOS_ID
                [5, 6, EOS_ID, 0],  # the content of chunk[3:4] padding after EOS_ID
            ],
            [[5, 6, EOS_ID, 0], [0, 0, 0, 0]],  # -1 means no knn
        ]  # retrieved 2 consecutive chunks from each of two knn results

        np.testing.assert_array_equal(actual_seq_tokens.numpy(), expected_seq_tokens)
        np.testing.assert_array_equal(actual_retrieved_tokens.numpy(), expected_retrieved_tokens, verbose=True)
