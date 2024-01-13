from joblib import delayed, Parallel
from typing import List
from omegaconf import DictConfig
from rich.progress import track
from pathlib import Path
import pickle as pkl
import numpy as np
import jsonlines
import traceback
import logging
import random
import torch
import faiss
import math
import time
import os
from tqdm import tqdm

# TODO: pull these out of retrieval and break connection to that codebase
import sys
sys.path.append("/fsx-instruct-opt/swj0419/rlm_pretrain/hcir/retro-z/retro_z/RETRO-pytorch")
from retro_pytorch.retrieval import tokenize, bert_embed
from chunk_logger import ChunkLogger, ChunkLoggerDummy


from submitit_utils import (
    WorkerFunctor,
    create_executor,
    await_completion_of_jobs,
    fetch_and_validate_neighbors_results,
    fetch_and_validate_embedding_results
)

from retro_z_utils import (
    reshape_memmap_given_width,
    exists_and_is_file,
    exists_and_is_dir,
    write_jsonl_file,
    read_jsonl_file,
    read_jsonl_file_no_compress,
    range_chunked,
    init_logging,
    memmap,
    log,
)

from retro_z_retrieval import (
    get_neighbors_and_continuations,
    get_last_chunk_of_doc_flags,
)
from ipdb import set_trace as bp

WORKERS_PER_FILE_WEIGHTINGS = {'CommonCrawl': 4,
                               'HackerNews': 0.125,
                               'Enron_Emails': 0.125,
                               'DM_Mathematics': 0.125,
                               'BookCorpusFair': 0.125}

random.seed(88)
np.random.seed(88)

# TODO: get these from the tokenizer
PAD_TOKEN = 0
UNK_TOKEN = 100
CLS_TOKEN = 101
SEP_TOKEN = 102
MASK_TOKEN = 103

EMBED_DIM = 768

JSONL_PAYLOAD = 'text'

GLOB_SEPARATOR = ','

NPY_GLOB = '*.npy'
LZ4_GLOB = '*.jsonl'
NPY_SUFFIX = '.npy'
LZ4_SUFFIX = '.lz4'
MAP_SUFFIX = '.map'
DIST_SUFFIX = '.dist'
LZ4_NPY_GLOB = '*.lz4.npy'

INDEX_FILENAME = 'index'
KNNS_FILENAME = 'knns.npy'
MAP_FILENAME = 'embeddings.map'
INDICES_FILENAME = 'embeddings.key'
CHUNKS_TO_DOCS_FILENAME = 'chunks_to_docs.npy'

NUM_CPUS_PER_NODE = 10

JOB_BATCH_SIZE = 1024
CHUNK_BATCH_SIZE = 1024

# >= 50 means print stdout
JOBLIB_VERBOSITY = 50

# swj
MAX_TOKENS=510

def _parallel(n_jobs=-1):
    # print("n_jobs: ", n_jobs)
    # swj change
    return Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSITY)
    # return Parallel(n_jobs=1, verbose=JOBLIB_VERBOSITY)



def _mask_after_eos(seq_tokens):
    assert seq_tokens is not None, 'Failed test: seq_tokens is not None'

    # mask out (with padding tokens) any token following an <eos> before the next sequence
    # after_eos_id = np.cumsum(seq_tokens == SEP_TOKEN, axis=-1, dtype=np.int32)
    # after_sos_id = np.cumsum(seq_tokens == CLS_TOKEN, axis=-1, dtype=np.int32)
    # seq_mask = np.array(after_eos_id ^ after_sos_id, dtype=np.bool)  # type: ignore
    seq_mask = (seq_tokens != PAD_TOKEN) & (seq_tokens != CLS_TOKEN)
    # TODO: don't need to return seq_tokens
    return seq_tokens, seq_mask


def _generate_one_example(chunks_slice_modeling_relative: slice, chunks_slice_modeling_absolute: slice, current_chunks: np.ndarray, includes_last_chunk: bool, knns: np.ndarray,
                          chunks_memmaps_retrieval, chunks_to_docs_retrieval: np.memmap, files_indices_retrieval, is_eval_dataset: bool, chunk_len: int, k_: int):
    assert chunks_slice_modeling_relative.start < chunks_slice_modeling_relative.stop, f'Invalid chunks_slice_modeling_relative: {chunks_slice_modeling_relative}'
    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'
    assert chunks_memmaps_retrieval is not None, 'Failed test: chunks_memmaps_retrieval is not None'
    assert files_indices_retrieval is not None, 'Failed test: files_indices_retrieval is not None'
    assert k_ > 0 and (k_ == 2 or k_ == 5), f'Invalid k_: {k_}'

    chunks = current_chunks[chunks_slice_modeling_relative]
    seq_tokens = chunks.flatten()
    assert len(seq_tokens == 2048), f'Invalid len(seq_tokens: {len(seq_tokens)}'

    last_token = PAD_TOKEN if includes_last_chunk else current_chunks[chunks_slice_modeling_relative.stop][0].item()
    target_tokens = np.concatenate((seq_tokens[1:], [last_token]), axis=-1, dtype=np.int32)

    seq_tokens, seq_mask = _mask_after_eos(seq_tokens)
    assert not (seq_tokens == PAD_TOKEN).all(), 'Invalid seq_tokens: all 0'

    neighbors_of_chunks = knns[chunks_slice_modeling_absolute, :]
    chunks_ids_absolute = np.arange(chunks_slice_modeling_absolute.start, chunks_slice_modeling_absolute.stop, dtype=np.uint32)

    indices_of_retrieval_memmaps = list(files_indices_retrieval.values())
    # swj: get neighbor and continuation tokens
    neighbor_tokens = get_neighbors_and_continuations(chunks_ids_absolute, neighbors_of_chunks, chunks_memmaps_retrieval, chunks_to_docs_retrieval,  # type:ignore
                                                      indices_of_retrieval_memmaps, chunk_len, k_, is_eval_dataset)
    last_chunk_of_doc_flags = get_last_chunk_of_doc_flags(chunks_ids_absolute, chunks_to_docs_retrieval)

    return {
        'example_tokens': seq_tokens,
        'example_mask': seq_mask,
        'target_tokens': target_tokens,
        'neighbor_tokens': neighbor_tokens,
        'last_chunk_of_doc_flags': last_chunk_of_doc_flags,
    }


def _determine_num_chunks_per_seq(seq_len: int, chunk_len: int):
    assert chunk_len > 0, f'Invalid chunk_len: {chunk_len}'
    assert seq_len > 0, f'Invalid seq_len: {seq_len}'
    num_chunks_per_seq, mod = divmod(seq_len, chunk_len)
    assert mod == 0, f'Invalid mod: {mod}'
    # assert num_chunks_per_seq == 32, f'Invalid num_chunks_per_seq: {num_chunks_per_seq}'

    return num_chunks_per_seq


def _generate_examples_from_memmap_slice(chunks_memmap: np.memmap, memmap_chunks_slice: slice, chunks_index: int,
                                         tfds_dir_path: Path, knns: np.memmap, chunks_memmaps: List[np.memmap],
                                         chunks_to_docs: np.memmap, files_indices, chunk_len: int, seq_len: int, k_: int,
                                         num_chunks_in_memmap: int):
    assert memmap_chunks_slice.start < memmap_chunks_slice.stop
    # assert chunk_len > 0 and chunk_len == 64
    # assert seq_len > 0 and seq_len == 2048
    assert files_indices is not None
    assert num_chunks_in_memmap > 0
    assert k_ > 0 and (k_ == 2 or k_ == 5)
    assert chunks_index >= 0

    num_chunks_per_seq = _determine_num_chunks_per_seq(seq_len, chunk_len)
    start_chunk, end_chunk = memmap_chunks_slice.start, memmap_chunks_slice.stop
    assert ((end_chunk - start_chunk) % num_chunks_per_seq) == 0

    # shard_index = memmap_chunks_slice.start // CHUNK_BATCH_SIZE
    # chunk_logger = ChunkLogger(chunk_len, seq_len, k_, tfds_dir_path, shard_index)

    min_chunk_index, _, _ = list(files_indices.items())[chunks_index]
    examples_list = []
    for example_chunks_slice in range_chunked(end_chunk, num_chunks_per_seq, exact=True, min_value=start_chunk):
        assert example_chunks_slice.stop <= num_chunks_in_memmap
        # includes_last_chunk = example_chunks_slice.stop == num_chunks_in_memmap
        # example_chunks_slice_absolute = slice(example_chunks_slice.start + min_chunk_index,
        #                                       example_chunks_slice.stop + min_chunk_index)
        breakpoint()
        # FIXME:
        # example = _generate_one_example(example_chunks_slice, example_chunks_slice_absolute, chunks_memmap,
        #                                 includes_last_chunk, knns, chunks_memmaps, chunks_to_docs,
        #                                 files_indices, chunk_len, k_)
        # chunk_logger.log_example(example)
        # examples_list.append(example)

    return examples_list


def _get_files_indices(embeddings_dir_path: Path):
    indices_file_path = embeddings_dir_path / INDICES_FILENAME
    logging.info(f'Loading files indices from {indices_file_path} and validating')

    assert exists_and_is_file(indices_file_path)
    with open(indices_file_path, 'rb') as files_indices_file:
        files_indices = pkl.load(files_indices_file)

    chunk_max_index = 0
    # FIXME:
    # prev_filename = ''
    prev_chunk_max_index = 0
    index = 0
    for filename, indices in files_indices.items():
        chunk_min_index, chunk_max_index, file_index = indices
        assert len(filename) > 0, 'Failed test: len(filename) > 0'
        # FIXME: decide whether to keep or remove this. 1g dataset did not sort
        # assert filename > prev_filename
        assert file_index == index, f'Invalid file_index ({file_index}) or index ({index})'
        assert chunk_min_index >= 0, f'Invalid chunk_min_index: {chunk_min_index}'
        assert chunk_max_index > chunk_min_index, f'Invalid chunk_min_index ({chunk_min_index}) or chunk_max_index ({chunk_max_index})'
        assert chunk_min_index == prev_chunk_max_index, f'Invalid chunk_min_index ({chunk_min_index}) or prev_chunk_max_index ({prev_chunk_max_index})'

        index += 1
        prev_chunk_max_index = chunk_max_index
        # FIXME:
        # prev_filename = filename

    return files_indices, chunk_max_index


# TODO: this could be done when generated chunks
def _create_aggregate_docs_map(tfds_dir_path: Path, chunks_dir_path: Path, files_indices, num_chunks_overall: int):
    assert files_indices is not None, 'Failed test: files_indices is not None'
    assert num_chunks_overall > 0, f'Invalid num_chunks_overall: {num_chunks_overall}'

    chunks_to_docs_filepath = tfds_dir_path / CHUNKS_TO_DOCS_FILENAME
    with log(f'Creating aggregate docs map at {chunks_to_docs_filepath}'):
        chunks_to_docs = np.memmap(chunks_to_docs_filepath, mode="w+", shape=(num_chunks_overall,), dtype=np.int32)

    document_offset = 0
    for filename, indices in files_indices.items():
        chunk_min_index, chunk_max_index, _ = indices
        file_path = chunks_dir_path / filename
        chunks_map_file_path = Path(str(file_path) + MAP_SUFFIX)

        logging.info(f'Copying contents of file {chunks_map_file_path} to aggregate map')
        with memmap(chunks_map_file_path, dtype=np.int32, mode='r') as chunks_map:
            assert len(chunks_map) == chunk_max_index - chunk_min_index, f'Invalid chunks map len ({len(chunks_map)}), chunk_max_index ({chunk_max_index}) or chunk_min_index ({chunk_min_index})'
            chunks_to_docs[chunk_min_index:chunk_max_index] = chunks_map + document_offset

            # TODO: simplify
            assert np.max(chunks_map) == chunks_map[-1], 'Failed test: np.max(chunks_map) == chunks_map[-1]'
            # +1 because the start of the next file will be a new document with a relative index of 0
            document_offset += np.max(chunks_map) + 1
    assert chunk_max_index == num_chunks_overall, f'Invalid chunk_max_index ({chunk_max_index}) or num_chunks_overall ({num_chunks_overall})'  # type: ignore

    return chunks_to_docs


def _get_aggregate_docs_map(tfds_dir_path: Path):
    chunks_to_docs_filepath = tfds_dir_path / CHUNKS_TO_DOCS_FILENAME
    return np.memmap(chunks_to_docs_filepath, mode='r', dtype=np.int32)


def _get_chunks_memmaps(chunks_dir_path: Path, files_indices, chunk_len: int):
    assert exists_and_is_dir(chunks_dir_path)
    assert files_indices is not None, 'Failed test: files_indices is not None'
    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'

    chunks_memmaps = [None] * len(files_indices)
    for filename, indices in files_indices.items():
        logging.info(f'Mapping chunks from file {filename} with indices {indices}')
        _, _, filename_index = indices
        file_path = chunks_dir_path / filename
        chunks_flat = np.memmap(file_path, np.int32, 'r')
        chunks, _ = reshape_memmap_given_width(chunks_flat, chunk_len)

        chunks_memmaps[filename_index] = chunks

    return chunks_memmaps


def _get_tfds_features(seq_len: int, chunk_len: int, k_: int):
    import tensorflow_datasets as tfds
    import tensorflow as tf

    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'
    # assert seq_len > 0 and seq_len == 2048, f'Invalid seq_len: {seq_len}'
    assert k_ > 0 and (k_ == 2 or k_ == 5), f'Invalid k_: {k_}'

    num_chunks_per_seq = _determine_num_chunks_per_seq(seq_len, chunk_len)

    features = {
        "example_tokens": tfds.features.Tensor(shape=(seq_len,), dtype=tf.int32),
        "example_mask": tfds.features.Tensor(shape=(seq_len,), dtype=tf.bool),
        "target_tokens": tfds.features.Tensor(shape=(seq_len,), dtype=tf.int32),
        # * 2 isn't for number of neighbors, but rather for the continuations
        "neighbor_tokens": tfds.features.Tensor(shape=(num_chunks_per_seq, k_, chunk_len * 2), dtype=tf.int32,),
        "last_chunk_of_doc_flags": tfds.features.Tensor(shape=(num_chunks_per_seq,), dtype=tf.bool,)}

    return tfds.features.FeaturesDict(features)  # type:ignore


def _add_ds_and_final_shard_to_filenames(tfds_dir_path: Path):
    # TODO: tighten this up
    file_paths = list(tfds_dir_path.glob('*.tfrecord-*'))
    num_shards = len(file_paths)
    for file_path in file_paths:
        filename = file_path.name
        file_path.rename(Path(os.path.dirname(file_path)) / Path(str(filename).replace('=', '_') + f'-of-{num_shards:>05d}'))

    # a long time to sleep but really not worth blowing up generation for this
    time.sleep(60 * 1)


def _write_tfds_metadata(tfds_dir_path: Path, shard_lengths: List[int], features, split: str):
    import tensorflow_datasets as tfds

    assert split in ['train', 'validation'], f'Invalid split: {split}'
    split_infos = [tfds.core.SplitInfo(name=split, shard_lengths=shard_lengths, num_bytes=0)]  # type: ignore

    with log('Writing tfds metadata'):
        tfds.folder_dataset.write_metadata(
            data_dir=f'{tfds_dir_path}',
            features=features,
            split_infos=split_infos,
            filename_template='{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}',
            # TODO:
            description='''Multi-line description.''',
            homepage='http://my-project.org',
            supervised_keys=('image', 'label'),
            citation='''BibTex citation.''')


def _write_tfds_records(tfds_dir_path: Path, shard_index: int, features, examples_list, split: str):
    import tensorflow as tf

    assert split in ['train', 'validation'], f'Invalid split: {split}'
    assert examples_list is not None, 'Failed test: examples_list is not None'
    assert shard_index >= 0, f'Invalid shard_index: {shard_index}'

    # tfds blows up if filename has - outside of template
    tfds_dataset_name = tfds_dir_path.parent.name.replace('-', '_')
    with log(f'Writing tfrecord for shard index {shard_index} with {len(examples_list)} examples'):
        tfrecord_path = tfds_dir_path / f'ds_{tfds_dataset_name}-{split}.tfrecord-{shard_index:>05d}'
        with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
            for example in examples_list:
                # TODO: move into generation
                example['neighbor_tokens'] = np.transpose(example['neighbor_tokens'], axes=[1, 0, 2])
                writer.write(features.serialize_example(example))


def _get_data_for_parallel_worker(modeling_chunks_index, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                  embeds_dir_path_modeling, embeds_dir_path_retrieval, tfds_dir_path, chunk_len, split):
    files_indices_modeling, chunks_memmaps_modeling, num_chunks_overall_modeling = _get_chunks_data_for_tfds_jobs(chunks_dir_path_modeling,
                                                                                                                  embeds_dir_path_modeling,
                                                                                                                  chunk_len)
    logging.info(f'Processing file {list(files_indices_modeling.keys())[modeling_chunks_index]}')

    if split == 'validation':
        assert chunks_dir_path_modeling != chunks_dir_path_retrieval, 'Failed test chunks_dir_path_modeling != chunks_dir_path_retrieval'
        assert embeds_dir_path_modeling != embeds_dir_path_retrieval, 'Failed test embeds_dir_path_modeling != embeds_dir_path_retrieval'

        files_indices_retrieval, chunks_memmaps_retrieval, _ = _get_chunks_data_for_tfds_jobs(chunks_dir_path_retrieval,
                                                                                              embeds_dir_path_retrieval,
                                                                                              chunk_len)
    else:
        assert chunks_dir_path_modeling == chunks_dir_path_retrieval, 'Failed test chunks_dir_path_modeling == chunks_dir_path_retrieval'
        assert embeds_dir_path_modeling == embeds_dir_path_retrieval, 'Failed test embeds_dir_path_modeling == embeds_dir_path_retrieval'

        chunks_memmaps_retrieval = chunks_memmaps_modeling
        files_indices_retrieval = files_indices_modeling

    chunks_to_docs_retrieval = _get_aggregate_docs_map(tfds_dir_path)

    return chunks_memmaps_modeling, num_chunks_overall_modeling, files_indices_retrieval, chunks_memmaps_retrieval, chunks_to_docs_retrieval


def _generate_and_write_tfds_examples_from_slice(shard_index, modeling_chunks_slice, modeling_chunks_index, knns_dir_path, tfds_dir_path, chunk_len, seq_len, k_, K,
                                                 num_chunks_in_memmap, min_chunk_index, split, chunks_memmaps_modeling, num_chunks_overall_modeling,
                                                 files_indices_retrieval, chunks_memmaps_retrieval, chunks_to_docs_retrieval, log_every_num_chunks: int):
    assert log_every_num_chunks != 0, 'log_every_num_chunks != 0'
    start_chunk, end_chunk = modeling_chunks_slice.start, modeling_chunks_slice.stop
    num_chunks_per_seq = _determine_num_chunks_per_seq(seq_len, chunk_len)
    assert end_chunk > start_chunk, f'Invalid start_chunks ({start_chunk}) or end_chunk ({end_chunk})'

    chunk_logger = ChunkLogger(chunk_len, seq_len, k_, tfds_dir_path, shard_index)

    chunks_memmap_modeling = chunks_memmaps_modeling[modeling_chunks_index]
    with memmap(knns_dir_path / KNNS_FILENAME, dtype=np.uint32, mode='r') as knns_flat:
        with log(f'Reshaping knns array from {knns_dir_path / KNNS_FILENAME}'):
            knns, num_knns_rows = reshape_memmap_given_width(knns_flat, K)
        assert num_chunks_overall_modeling == num_knns_rows, f'{num_chunks_overall_modeling} == {num_knns_rows}'

        examples_list = []
        for index, example_chunks_slice in enumerate(range_chunked(end_chunk, num_chunks_per_seq, exact=True, min_value=start_chunk)):
            assert example_chunks_slice.stop <= num_chunks_in_memmap, f'Invalid example_chunks_slice({example_chunks_slice}) or num_chunks_in_memmap ({num_chunks_in_memmap})'
            includes_last_chunk = example_chunks_slice.stop == num_chunks_in_memmap
            example_chunks_slice_absolute = slice(example_chunks_slice.start + min_chunk_index,
                                                  example_chunks_slice.stop + min_chunk_index)
            logging.debug(f'Generating example for absolute slice {example_chunks_slice_absolute} '
                          f'and relative slice {example_chunks_slice}')

            example = _generate_one_example(example_chunks_slice, example_chunks_slice_absolute, chunks_memmap_modeling, includes_last_chunk, knns,  # type: ignore
                                            chunks_memmaps_retrieval, chunks_to_docs_retrieval, files_indices_retrieval, split == 'validation', chunk_len, k_)
            if (index % log_every_num_chunks) == 0:
                chunk_logger.log_example(example, flush=False)
            examples_list.append(example)

    features = _get_tfds_features(seq_len, chunk_len, k_)
    _write_tfds_records(tfds_dir_path, shard_index, features, examples_list, split)

    return len(examples_list)


def _tfds_parallel_cpu_worker(shard_index: int, modeling_chunks_slice: slice, modeling_chunks_index: int,
                              chunks_dir_path_modeling: Path, chunks_dir_path_retrieval: Path,
                              embeds_dir_path_modeling: Path, embeds_dir_path_retrieval: Path, knns_dir_path: Path,
                              tfds_dir_path: Path, chunk_len: int, seq_len: int, k_: int, K: int,  # noqa: N803
                              num_chunks_in_memmap: int, min_chunk_index: int, split: str, log_every_num_chunks: int):
    init_logging()
    # bp()
    try:
        logging.info('CPU worker checking parameters')

        # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'
        assert split in ['train', 'validation'], f'Invalid split: {split}'
        # assert seq_len > 0 and seq_len == 2048, f'Invalid seq_len: {seq_len}'
        assert modeling_chunks_index >= 0, f'Invalid modeling_chunks_index: {modeling_chunks_index}'
        assert num_chunks_in_memmap > 0, f'Invalid num_chunks_in_memmap: {num_chunks_in_memmap}'
        assert k_ > 0 and (k_ == 2 or k_ == 5), f'Invalid k_: {k_}'
        assert K > k_ and K == 50, f'Invalid K: {K}'
        assert shard_index >= 0, f'Invalid shard_index: {shard_index}'
        assert log_every_num_chunks != 0, 'log_every_num_chunks != 0'

        logging.info(f'CPU worker processing shard index {shard_index}, slice ({modeling_chunks_slice.start}, {modeling_chunks_slice.stop})')

        chunks_memmaps_modeling, num_chunks_overall_modeling, files_indices_retrieval, chunks_memmaps_retrieval, chunks_to_docs_retrieval = \
            _get_data_for_parallel_worker(modeling_chunks_index, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                          embeds_dir_path_modeling, embeds_dir_path_retrieval, tfds_dir_path, chunk_len, split)
        # bp()
        examples_len = _generate_and_write_tfds_examples_from_slice(shard_index, modeling_chunks_slice, modeling_chunks_index, knns_dir_path,
                                                                    tfds_dir_path, chunk_len, seq_len, k_, K, num_chunks_in_memmap,
                                                                    min_chunk_index, split, chunks_memmaps_modeling, num_chunks_overall_modeling,
                                                                    files_indices_retrieval, chunks_memmaps_retrieval, chunks_to_docs_retrieval,
                                                                    log_every_num_chunks)
    except Exception as ex:
        logging.critical(f'Error in CPU worker:\n{ex}')
        logging.critical(traceback.format_exc())
        raise ex

    return examples_len


def _create_num_cpus_chunk_aligned_batches(modeling_chunks_slice: slice, num_chunks_per_seq: int):
    node_slice_length = modeling_chunks_slice.stop - modeling_chunks_slice.start
    divisor = NUM_CPUS_PER_NODE * num_chunks_per_seq
    # we round up here and will adjust our final batch size downwards if necessary, below
    seqs_per_node = math.ceil(node_slice_length / divisor)
    batches = list(range_chunked(modeling_chunks_slice.stop, seqs_per_node * num_chunks_per_seq, min_value=modeling_chunks_slice.start))
    final_batch_sz = batches[-1].stop - batches[-1].start
    adj_final_stop = _round_to_multiple(final_batch_sz, num_chunks_per_seq, 'down')
    batches[-1] = slice(batches[-1].start, batches[-1].start + adj_final_stop)

    return batches


# TODO: check params (inside try)
def _tfds_parallel_node_worker(shard_index: int, modeling_chunks_slice: slice, modeling_chunks_index: int,
                               chunks_dir_path_modeling: Path, chunks_dir_path_retrieval: Path,
                               embeds_dir_path_modeling: Path, embeds_dir_path_retrieval: Path, knns_dir_path: Path,
                               tfds_dir_path: Path, chunk_len: int, seq_len: int, k_: int, K: int,  # noqa: N803
                               num_chunks_in_memmap: int, min_chunk_index: int, split: str, log_every_num_chunks: int):
    results = []
    try:
        num_chunks_per_seq = _determine_num_chunks_per_seq(seq_len, chunk_len)
        batches = _create_num_cpus_chunk_aligned_batches(modeling_chunks_slice, num_chunks_per_seq)
        logging.info(f'Processing batches {batches}')

        assert len(batches) <= NUM_CPUS_PER_NODE, f'Number of CPUs {NUM_CPUS_PER_NODE} != number of jobs {len(batches)}'
        # if the sub-slice length is less than 320 (what's required to 32-align on 10 CPUs) i.e. overall slice length is < 3200
        # we won't the optimal number of jobs
        if len(batches) < NUM_CPUS_PER_NODE:
            logging.warning(f'Number of CPUs {NUM_CPUS_PER_NODE} < number of jobs {len(batches)}. Probably slice length is < 320 * 10 ({modeling_chunks_slice.stop - modeling_chunks_slice.start}).')

        results = _parallel(n_jobs=NUM_CPUS_PER_NODE)(delayed(_tfds_parallel_cpu_worker)(shard_index + node_index, cpu_slice, modeling_chunks_index,
                                                                                         chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                                                                         embeds_dir_path_modeling, embeds_dir_path_retrieval,
                                                                                         knns_dir_path, tfds_dir_path, chunk_len, seq_len, k_, K,
                                                                                         num_chunks_in_memmap, min_chunk_index, split, log_every_num_chunks)
                                                      for node_index, cpu_slice in enumerate(batches))
    except Exception as ex:
        logging.critical(f'Error in node worker:\n{ex}')
        logging.critical(traceback.format_exc())
        raise ex

    return results


def _round_to_multiple(number: int, multiple: int, direction: str):
    assert direction in ['up', 'down'], f'Invalid direction: {direction}'
    if direction == 'up':
        return multiple * math.ceil(number / multiple)
    else:
        return multiple * math.floor(number / multiple)


def _submit_tfds_jobs_for_chunks_memmap(executor, base_shard_index: int, tfds_dir_path: Path, chunks_dir_path_modeling: Path, chunks_dir_path_retrieval: Path,
                                        embeds_dir_path_modeling: Path, embeds_dir_path_retrieval: Path, knns_dir_path: Path, chunk_len: int, seq_len: int,
                                        k_: int, K: int, num_workers_per_file: int, chunks_filename: str, indices, split: str, log_every_num_chunks: int):  # noqa: N803
    min_chunk_index, max_chunk_index, chunks_memmap_index = indices

    assert max_chunk_index > min_chunk_index
    # assert chunk_len > 0 and chunk_len == 64
    assert split in ['train', 'validation']
    # assert seq_len > 0 and seq_len == 2048
    assert len(chunks_filename) > 0
    assert num_workers_per_file > 0
    assert chunks_memmap_index >= 0
    assert base_shard_index >= 0
    assert min_chunk_index >= 0
    assert k_ > 0 and (k_ == 2 or k_ == 5)
    assert K > k_
    assert log_every_num_chunks > 0

    weighted_workers = num_workers_per_file
    for substr, weighting in WORKERS_PER_FILE_WEIGHTINGS.items():
        if substr in chunks_filename:
            weighted_workers = int(np.ceil(num_workers_per_file * weighting))
            break
    logging.info(f'Distributing to weighted number of workers: {weighted_workers}')

    num_chunks_in_memmap = max_chunk_index - min_chunk_index
    batch_size = _round_to_multiple(max_chunk_index / weighted_workers, 1, 'up')
    logging.info(f'Processing from chunk {min_chunk_index} to {max_chunk_index} with batch size {batch_size}')

    shard_index = base_shard_index
    jobs = []

    for chunks_slice in range_chunked(num_chunks_in_memmap, batch_size):
        if (chunks_slice.stop - chunks_slice.start) * _determine_num_chunks_per_seq(seq_len, chunk_len) < NUM_CPUS_PER_NODE:
            logging.info(f'Skipping slice {chunks_slice} as too small for #CPUS')
        else:
            logging.info(f'Submitting job on slice {chunks_slice} with shard index {shard_index}')
            worker_fn = WorkerFunctor(_tfds_parallel_node_worker, shard_index, chunks_slice,
                                      chunks_memmap_index, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                      embeds_dir_path_modeling, embeds_dir_path_retrieval, knns_dir_path,
                                      tfds_dir_path, chunk_len, seq_len, k_, K, num_chunks_in_memmap, min_chunk_index,
                                      split, log_every_num_chunks)
            job = executor.submit(worker_fn)
            jobs.append(job)
            shard_index += NUM_CPUS_PER_NODE

    logging.info(f'Submitted {len(jobs)} jobs for file')
    assert shard_index - base_shard_index == len(jobs) * NUM_CPUS_PER_NODE
    return jobs, shard_index - base_shard_index


# TODO: wildly refactor this function and caller
def _get_chunks_data_for_tfds_jobs(chunks_dir_path, embeddings_dir_path, chunk_len):
    with log('Loading files indices'):
        files_indices, num_chunks_overall = _get_files_indices(embeddings_dir_path)

    chunks_memmaps = _get_chunks_memmaps(chunks_dir_path, files_indices, chunk_len)

    return files_indices, chunks_memmaps, num_chunks_overall


def chunks_and_knns_to_tfds_dataset_parallel(tfds_dir_path: Path, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                             embeds_dir_path_modeling, embeds_dir_path_retrieval,
                                             knns_dir_path: Path, chunk_len: int, seq_len: int, K: int, k_: int,  # noqa: N803
                                             parallel_cfg: DictConfig, split: str, log_every_num_chunks: int):
    assert exists_and_is_dir(chunks_dir_path_retrieval)
    assert exists_and_is_dir(embeds_dir_path_retrieval)
    assert exists_and_is_dir(chunks_dir_path_modeling)
    assert exists_and_is_dir(embeds_dir_path_modeling)
    assert exists_and_is_dir(tfds_dir_path)
    assert exists_and_is_dir(knns_dir_path)
    # assert chunk_len > 0 and chunk_len == 64
    assert split in ['train', 'validation']
    # assert seq_len > 0 and seq_len == 2048
    assert k_ > 0 and (k_ == 2 or k_ == 5)
    assert K > k_ and K == 50
    assert log_every_num_chunks > 0

    files_indices_modeling, chunks_memmaps_modeling, num_chunks_overall_modeling = _get_chunks_data_for_tfds_jobs(chunks_dir_path_modeling, embeds_dir_path_modeling, chunk_len)

    if split == 'validation':
        assert chunks_dir_path_modeling != chunks_dir_path_retrieval
        assert embeds_dir_path_modeling != embeds_dir_path_retrieval
        files_indices_retrieval, _, num_chunks_overall_retrieval = _get_chunks_data_for_tfds_jobs(chunks_dir_path_retrieval, embeds_dir_path_retrieval, chunk_len)
    else:
        assert chunks_dir_path_modeling == chunks_dir_path_retrieval
        assert embeds_dir_path_modeling == embeds_dir_path_retrieval
        files_indices_retrieval, num_chunks_overall_retrieval = files_indices_modeling, num_chunks_overall_modeling

    with log('Creating aggregate docs map'):
        chunks_to_docs = _create_aggregate_docs_map(tfds_dir_path, chunks_dir_path_retrieval, files_indices_retrieval, num_chunks_overall_retrieval)
        assert len(chunks_to_docs) == num_chunks_overall_retrieval

    all_jobs = []
    base_shard_index, total_shards = 0, 0
    # bp()
    executor = create_executor(parallel_cfg.submitit)
    with executor.batch():
        logging.info(f'Submitting jobs for {len(chunks_memmaps_modeling)} chunks files')  # type:ignore
        for chunks_filename, indices in files_indices_modeling.items():  # type:ignore
            logging.info(f'Submitting jobs for {chunks_filename} with base shard index {base_shard_index} and indices {indices}')
            memmap_jobs, total_shards = _submit_tfds_jobs_for_chunks_memmap(executor, base_shard_index, tfds_dir_path, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                                                            embeds_dir_path_modeling, embeds_dir_path_retrieval, knns_dir_path, chunk_len, seq_len, k_, K,
                                                                            parallel_cfg.num_workers_per_file, chunks_filename, indices, split, log_every_num_chunks)
            all_jobs.extend(memmap_jobs)
            base_shard_index += total_shards
    logging.info(f'Submitted {len(all_jobs)} jobs')

    try:
        with log('Waiting for jobs to complete'):
            await_completion_of_jobs(all_jobs)
        with log('Fetching and validating results'):
            shard_lengths = fetch_and_validate_neighbors_results(all_jobs)
            features = _get_tfds_features(seq_len, chunk_len, k_)
            _add_ds_and_final_shard_to_filenames(tfds_dir_path)
            _write_tfds_metadata(tfds_dir_path, shard_lengths, features, split)
    except Exception as e:
        logging.critical(f'Error while awaiting/fetching/merging results. Cannot write metadata:\n{e}')


# TODO: dispose of chunks memmaps and chunks_to_docs (as well as underlying storage)
# TODO: probably batching doesn't achieve anything in serial?
# TODO: more logging, here and in called functions
def chunks_and_knns_to_tfds_dataset_serial(tfds_dir_path: Path, chunks_dir_path: Path, embeddings_dir_path: Path,
                                           knns_dir_path: Path, chunk_len: int, seq_len: int, K: int, k_: int, split: str):  # noqa: N803
    # TODO: update this function to log critical on failure
    assert exists_and_is_dir(chunks_dir_path); assert exists_and_is_dir(tfds_dir_path)
    assert exists_and_is_dir(knns_dir_path)
    # assert chunk_len > 0 and chunk_len == 64
    # assert seq_len > 0 and seq_len == 2048
    assert split in ['train', 'validation']
    assert k_ > 0 and (k_ == 2 or k_ == 5)
    assert K > k_ and K == 50

    num_chunks_per_seq, mod = divmod(seq_len, chunk_len)
    assert mod == 0

    files_indices, num_chunks_overall = _get_files_indices(embeddings_dir_path)

    with memmap(knns_dir_path / KNNS_FILENAME, dtype=np.int32, mode='r') as knns_flat:
        knns, num_knns_rows = reshape_memmap_given_width(knns_flat, K)
        assert num_chunks_overall == num_knns_rows
        chunks_memmaps, = _get_chunks_memmaps(chunks_dir_path, files_indices, chunk_len)
        chunks_to_docs = _create_aggregate_docs_map(tfds_dir_path, chunks_dir_path, files_indices, num_chunks_overall)

        with log(f'Processing {len(chunks_memmaps)} chunks files'):  # type: ignore
            examples_list = []
            for chunks_index, chunks_memmap in enumerate(chunks_memmaps):  # type: ignore
                final_chunk = _round_to_multiple(len(chunks_memmap), num_chunks_per_seq, 'down')  # type: ignore
                for chunks_slice in range_chunked(final_chunk, CHUNK_BATCH_SIZE):
                    examples = _generate_examples_from_memmap_slice(chunks_memmap, chunks_slice, chunks_index, tfds_dir_path,
                                                                    knns, chunks_memmaps, chunks_to_docs,  # type: ignore
                                                                    files_indices, chunk_len, seq_len, k_, num_chunks_overall)
                    examples_list.extend(examples)

    features = _get_tfds_features(seq_len, chunk_len, k_)
    _write_tfds_records(tfds_dir_path, 0, features, examples_list, split)
    _add_ds_and_final_shard_to_filenames(tfds_dir_path)
    _write_tfds_metadata(tfds_dir_path, [len(examples_list)], features, split)


def _convert_to_gpu_index(cpu_index: faiss.Index) -> faiss.Index:
    # convert an index to an gpu index that employs all machine gpus
    assert faiss.get_num_gpus() > 0
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    # fp16 exact search provides very accurate results
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)

    return gpu_index


# TODO: parameterize writing of distances
def _embeds_file_plus_index_to_cached_knns(index, file_start_index, embeddings_file_path: Path,
                                           embed_dim: int, k_: int, batch_size: int, knns, distances):
    assert exists_and_is_file(embeddings_file_path)
    assert distances is not None
    assert knns is not None
    assert batch_size > 0
    assert embed_dim > 0
    assert k_ > 0
    assert index

    with memmap(embeddings_file_path, np.float32, 'r') as embeds_flat:
        embeddings, num_embeds = reshape_memmap_given_width(embeds_flat, embed_dim)

        for embed_slice in range_chunked(num_embeds, batch_size):
            with log(f'Calculating knns {embed_slice.start} / {num_embeds}'):
                relative_query_indices = np.arange(embed_slice.start, embed_slice.stop)
                queries = embeddings[relative_query_indices]
                # swj: search for neighbors 
                neighbor_distances, absolute_neighbor_indices = index.search(queries, k_)
                absolute_query_indices = relative_query_indices + file_start_index
                # bp()
                knns[absolute_query_indices] = absolute_neighbor_indices
                distances[absolute_query_indices] = neighbor_distances


def embeds_dir_plus_index_to_cached_knns(index_reference: bool, index_dir_path: Path, embeddings_dir_path: Path,
                                         knns_dir_path: Path, batch_size: int, embed_dim: int, k_: int, enable_gpu: bool):
    assert not index_reference or exists_and_is_dir(knns_dir_path)
    assert exists_and_is_dir(index_dir_path)
    assert exists_and_is_dir(embeddings_dir_path)
    assert batch_size > 0
    assert embed_dim > 0
    assert k_ > 0

    index_file_path = index_dir_path / INDEX_FILENAME
    knns_file_path = knns_dir_path / KNNS_FILENAME

    with log('Reading index and key file'):
        index = faiss.read_index(str(index_file_path))
        if enable_gpu:
            index = _convert_to_gpu_index(index)

        files_indices, _ = _get_files_indices(embeddings_dir_path)

    # TODO: we don't need index_map really if we're calculating a reference index, *except* for for this length
    #       probably we can break this dependency
    with memmap(embeddings_dir_path / MAP_FILENAME, np.uint16, 'r') as index_map:
        knns_shape = (len(index_map), k_)
        with memmap(knns_file_path, np.uint32, 'w+', shape=knns_shape) as knns, \
             memmap(str(knns_file_path) + DIST_SUFFIX, np.float32, 'w+', shape=knns_shape) as distances:
            embeddings_file_paths = sorted(list(embeddings_dir_path.glob(NPY_GLOB)))
            with log(f'Processing {len(embeddings_file_paths)} embeddings files'):
                # swj: iterate over every embedding files 
                for embeddings_file_path in embeddings_file_paths:
                    # bp()
                    file_start_index, _, _ = files_indices[embeddings_file_path.name]
                    _embeds_file_plus_index_to_cached_knns(index, file_start_index, embeddings_file_path,
                                                           embed_dim, k_, batch_size, knns, distances)


# TODO: - most of this logic could be done when creating embeddings, not when creating the index
#       - per above, could also remove index map
def embeds_dir_to_index_swj(embeddings_dir_path: Path, index_dir_path: Path, index_key, embed_dim: int):
    assert exists_and_is_dir(index_dir_path)
    assert embeddings_dir_path.exists()
    assert index_key

    index_file_path = index_dir_path / INDEX_FILENAME
    print("index_file_path: ", index_file_path)
    logging.info(f'Faiss version: {faiss.__version__}; number of GPUs: {faiss.get_num_gpus()}')

    with log('Processing embeddings'):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        # embeddings_file_paths.reverse()
        embeddings_file_paths = [sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))[0]]
        for embeddings_file_path in embeddings_file_paths:
            with log(f'Processing embeddings file {embeddings_file_path.name}'), memmap(embeddings_file_path, np.float32, 'r') as embeds_flat:
                embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)               
                # train embeddings 
                train_index_path = train_index(embeddings, index_file_path)

        # add embeddings to faiss
        index = faiss.read_index(train_index_path)
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        with log(f'Processing {len(embeddings_file_paths)} embeddings files'):
            for embeddings_file_path in embeddings_file_paths:
                with log(f'Processing embeddings file {embeddings_file_path.name}'), \
                     memmap(embeddings_file_path, np.float32, 'r') as embeds_flat:
                    embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)
                    with log('Adding embeddings to index'):
                        index.add(embeddings.astype('float32'))  # type: ignore

    with log('Writing index'):
        faiss.write_index(index, str(index_file_path))

def train_index(embeddings, index_file_path):
    dimension=768
    ncentroids=4096
    code_size=64
    probe = 8
    cuda=1
    output_path = str(index_file_path) + "/index.trained"
    if not os.path.exists(output_path):
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(dimension)

        start_index = faiss.IndexIVFPQ(quantizer, dimension, ncentroids, code_size, 8)
        start_index.nprobe = probe

        print('Training Index')
        np.random.seed(0)
        start = time.time()

        if cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False
            # Train on GPU and back to CPU
            gpu_index.train(embeddings)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(embeddings)
            print('Training took {} s'.format(time.time() - start))

        print('Writing index after training')
        start = time.time()
        faiss.write_index(start_index, output_path)
        print('Writing index took {} s'.format(time.time() - start))
    return output_path

# TODO: - most of this logic could be done when creating embeddings, not when creating the index
#       - per above, could also remove index map
def embeds_dir_to_index(embeddings_dir_path: Path, index_dir_path: Path, index_key, embed_dim: int):
    assert exists_and_is_dir(index_dir_path)
    assert embeddings_dir_path.exists()
    assert index_key

    index_file_path = index_dir_path / INDEX_FILENAME

    logging.info(f'Faiss version: {faiss.__version__}; number of GPUs: {faiss.get_num_gpus()}')
    index = faiss.IndexFlat(EMBED_DIM, faiss.METRIC_L2)

    with log('Processing embeddings'):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        # embeddings_file_paths.reverse()
        with log(f'Processing {len(embeddings_file_paths)} embeddings files'):
            for embeddings_file_path in embeddings_file_paths:
                with log(f'Processing embeddings file {embeddings_file_path.name}'), \
                     memmap(embeddings_file_path, np.float32, 'r') as embeds_flat:

                    embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)
                    with log('Adding embeddings to index'):
                        # bp()
                        # maybe out of memory
                        index.add(embeddings.astype('float32'))  # type: ignore
                        with log(f'Writing index for {embeddings_file_path.name} in {str(index_dir_path)}'):
                            faiss.write_index(index, str(index_file_path))

    # with log('Writing index'):
    #     faiss.write_index(index, str(index_file_path))


# TODO: possibly overriding below todo, this could be done in chunk-creation
# TODO: this could be done while creating embeddings (not as a separate step afterwards)
#      we would just need to gather data back after parallel processing
#
def _create_map_and_key(embeddings_dir_path: Path, embed_dim: int):
    # bp()
    with log('Processing embeddings'):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(NPY_GLOB)))
        # print("embeddings_file_paths: ", embeddings_file_paths)
        with log(f'Processing {len(embeddings_file_paths)} embeddings files'):
            file_end_index = 0
            file_end_index_to_filename = {}
            for embeddings_file_path in embeddings_file_paths:
                with log(f'Processing embeddings file {embeddings_file_path.name}'), \
                     memmap(embeddings_file_path, np.float32, 'r') as embeds_flat:

                    _, num_embeds = reshape_memmap_given_width(embeds_flat, embed_dim)
                    file_end_index += num_embeds
                    file_end_index_to_filename[file_end_index] = embeddings_file_path.name

    with log('Writing map file'):
        filename_to_indices = {}
        # bp()
        with memmap(embeddings_dir_path / MAP_FILENAME, np.uint16, 'w+', shape=(file_end_index,)) as index_map:
            assert len(file_end_index_to_filename) <= 65536, 'Exceeded max # files of 65536 (file end indices stored in uint16)'
            file_start_index = 0
            filename_index = 0
            # relies on python dictionaries being ordered (which, since 3.6/3.7 they are)
            for file_end_index, filename in file_end_index_to_filename.items():
                index_map[file_start_index: file_end_index] = filename_index
                filename_to_indices[filename] = (file_start_index, file_end_index, filename_index)
                filename_index += 1
                file_start_index = file_end_index

    with log('Writing key file'):
        with open(embeddings_dir_path / INDICES_FILENAME, 'wb') as key_file:
            pkl.dump(filename_to_indices, key_file)


def _embed_chunk_batch(chunk_batch, model):
    # bp()
    # weijia hard code
    # chunk_batch = chunk_batch[:, :510]
    padded_batch = np.concatenate((chunk_batch, np.full((chunk_batch.shape[0], 2), PAD_TOKEN)), axis=1)
    padded_batch_ori = padded_batch.copy()

    for index, _ in enumerate(padded_batch):
        if padded_batch[index, 0] != CLS_TOKEN:
            padded_batch[index] = np.roll(padded_batch[index], 1)
            padded_batch[index, 0] = CLS_TOKEN

        if SEP_TOKEN not in padded_batch[index]:
            pad_indices = np.where(padded_batch[index] == PAD_TOKEN)
            assert len(pad_indices) == 1
            pad_indices = pad_indices[0]
            padded_batch[index, pad_indices[0]] = SEP_TOKEN
    # bp()
    # print("padded_batch: ", padded_batch)
    padded_batch_torch = torch.from_numpy(padded_batch)
    assert padded_batch_torch.shape[1] <= 512

    # bp()
    batch_embed = bert_embed(padded_batch_torch, model_config=model)
    return batch_embed


def _parallel_embed(chunks_file_path: Path, embeds_file_path: Path, batch_size: int, num_workers: int,
                    worker_id: int, model: DictConfig, chunk_len: int):
    assert exists_and_is_file(chunks_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    assert num_workers > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert model

    with memmap(chunks_file_path, np.int32, 'r') as chunks_flat:
        num_chunks, mod = divmod(len(chunks_flat), chunk_len)
        assert num_chunks > 0 and mod == 0
        chunks = chunks_flat.reshape(num_chunks, chunk_len)

        shard_size = math.ceil(num_chunks / num_workers)
        start, end = shard_size * worker_id, shard_size * (worker_id + 1)
        shard = chunks[start:end]

        emb_list = []
        with log(f'Worker {worker_id} processing chunks {start} to {end}'):
            for row in range(0, shard_size, batch_size):
                batch_chunk_npy = shard[row:row + batch_size]
                batch_embed = _embed_chunk_batch(batch_chunk_npy, model)
                emb_list.append(batch_embed.detach().cpu().numpy())
            embeddings = np.vstack(emb_list)

        with log(f'Worker {worker_id} writing embeddings'):
            shard_filename = Path(str(embeds_file_path) + f'_{worker_id}_{num_workers}.npy')
            np.save(shard_filename, embeddings)

    return num_chunks, embeddings.shape[0]


def _parallel_chunks_file_to_embed_file(chunks_file_path: Path, embeds_file_path: Path, batch_size: int, model: DictConfig,
                                        chunk_len: int, embed_dim: int, parallel_cfg: DictConfig):
    assert exists_and_is_file(chunks_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    assert embed_dim > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert parallel_cfg
    assert model
    print("asdasdsa")
    with log(f'Processing file {chunks_file_path.name}'):
        num_workers = parallel_cfg.num_workers

        with log(f'Submitting {num_workers} jobs'):
            executor = create_executor(parallel_cfg.submitit)
            jobs = []
            with executor.batch():
                for worker_id in range(num_workers):
                    worker_fn = WorkerFunctor(_parallel_embed, chunks_file_path, embeds_file_path,
                                              batch_size, num_workers, worker_id, model, chunk_len)
                    job = executor.submit(worker_fn)
                    jobs.append(job)

        try:
            with log('Waiting for jobs to complete'):
                await_completion_of_jobs(jobs)
            with log('Fetching and validating results'):
                processed_chunks = fetch_and_validate_embedding_results(jobs)

            with log('Merging shards'):
                with memmap(embeds_file_path, np.float32, 'w+', shape=(processed_chunks, embed_dim)) as embeds:
                    embeds_index = 0
                    for worker_id in track(range(num_workers)):
                        shard_filename = Path(str(embeds_file_path) + f'_{worker_id}_{num_workers}.npy')
                        embeds_shard = np.load(shard_filename)
                        embeds[embeds_index: embeds_index + len(embeds_shard)] = embeds_shard
                        embeds_index += len(embeds_shard)
                        shard_filename.unlink()
        except Exception as e:
            logging.critical(f'Error while awaiting/fetching/merging results in file {chunks_file_path.name}:\n{e}')


def parallel_chunks_files_to_embeds_files(chunks_dir_path: Path, embeds_dir_path: Path, batch_size: int, model: DictConfig,
                                          embed_dim: int, chunk_len: int, parallel_cfg: DictConfig):
    assert exists_and_is_dir(chunks_dir_path)
    assert exists_and_is_dir(embeds_dir_path)
    assert batch_size > 0
    assert embed_dim > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert parallel_cfg
    assert model
    # print("chunks_dir_path: ", chunks_dir_path)
    # print(list(chunks_dir_path.glob(NPY_GLOB)))
    chunks_file_paths = sorted(list(chunks_dir_path.glob(NPY_GLOB)))

    # # swj hard code: filter out the files that have been processed
    # new_chunks_file_paths = []
    # for chunks_file_path in chunks_file_paths:
    #     for id in range(23, 24):
    #         if str(id) in chunks_file_path.name:
    #             new_chunks_file_paths.append(chunks_file_path)
    #             continue
    chunks_file_paths = chunks_file_paths
    # print("chunks_file_paths: ", chunks_file_paths)
    # bp()
    with log(f'Processing {len(chunks_file_paths)} chunks files'):
        for chunks_file_path in chunks_file_paths:
            _parallel_chunks_file_to_embed_file(chunks_file_path, embeds_dir_path / chunks_file_path.name,
                                                batch_size, model, chunk_len, embed_dim, parallel_cfg)

    # swj change
    # _create_map_and_key(embeds_dir_path, embed_dim)


def _chunks_file_to_embed_file(chunks_file_path: Path, embeds_file_path: Path, model: DictConfig,
                               chunk_len: int, batch_size: int, embed_dim: int):
    assert exists_and_is_file(chunks_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    assert embed_dim > 0
    assert model

    with log(f'Processing {chunks_file_path.name}'):
        with memmap(chunks_file_path, np.int32, 'r') as chunks_flat:
            # bp()
            chunks, num_chunks = reshape_memmap_given_width(chunks_flat, chunk_len)

            with memmap(embeds_file_path, np.float32, 'w+', shape=(num_chunks, embed_dim)) as embeds:
                for slice_ in range_chunked(num_chunks, batch_size):
                    chunk_batch = chunks[slice_]
                    embed_batch = _embed_chunk_batch(chunk_batch, model)
                    embeds[slice_] = embed_batch.cpu()


def chunks_files_to_embeds_files(chunks_dir_path: Path, embeds_dir_path: Path, model: DictConfig,
                                 chunk_len: int, batch_size: int, embed_dim: int):
    assert exists_and_is_dir(chunks_dir_path)
    assert exists_and_is_dir(embeds_dir_path)
    assert batch_size > 0
    assert embed_dim > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert model

    chunks_file_paths = sorted(list(chunks_dir_path.glob(NPY_GLOB)))
    # bp()
    with log(f'Processing {len(chunks_file_paths)} chunks files'):
        for chunks_file_path in chunks_file_paths:
            _chunks_file_to_embed_file(chunks_file_path, embeds_dir_path / chunks_file_path.name,
                                       model, chunk_len, batch_size, embed_dim)

    _create_map_and_key(embeds_dir_path, embed_dim)


def _create_chunks_and_map(tokens_file_path: Path, chunks_file_path: Path, chunk_len: int, total_chunks: int):
    assert exists_and_is_file(tokens_file_path)
    assert not chunks_file_path.is_dir()
    print("tokens_file_path: ", tokens_file_path, "total_chunks: ", total_chunks)
    # assert total_chunks > 0
    # assert chunk_len > 0 and chunk_len == 64

    with read_jsonl_file(tokens_file_path) as tokens_reader, \
         memmap(chunks_file_path, np.int32, 'w+', shape=(total_chunks, chunk_len)) as chunks, \
         memmap(str(chunks_file_path) + MAP_SUFFIX, np.int32, 'w+', shape=(total_chunks,)) as chunks_map:
        print("tokens_file_path: ", tokens_file_path)
        chunk_index = 0
        # each line in tokens file maps to equivalent line in documents file and so line index == doc_index
        '''
        for doc_index, line in enumerate(tokens_reader)
            print(line)
            if doc_index > 3:
                break
        '''
        for doc_index, line in enumerate(tokens_reader):
            # print(line["doc_id"])
            # bp()
            # assert doc_index == line["doc_id"]
            # pad to end of chunk
            tokens = line['tokens']
            # bp()
            # if len(tokens) <= 2:
            #     # TODO this is a bit duplicative of "discarding" check/log below - remove it when confident
            #     logging.warn(f'No tokens found while processing doc {doc_index} in file {tokens_file_path.name} - not discarding')
            # bp()
            div, mod = divmod(len(tokens), chunk_len)
            if mod != 0:
                tokens.extend([PAD_TOKEN] * (chunk_len - mod))
                div += 1
            if div != 1:
                bp()
            assert div == 1
            for doc_chunk_index in range(div):
                tokens_slice = slice(doc_chunk_index * chunk_len, (doc_chunk_index + 1) * chunk_len)
                chunks[chunk_index] = tokens[tokens_slice]
                chunks_map[chunk_index] = doc_index
                chunk_index += 1
            # print("chunk_index: ", chunk_index, "doc_id: ", line["doc_id"])
            # assert chunk_index == line["doc_id"]+1
    assert chunk_index == total_chunks


def _tokens_file_to_chunks_files(tokens_file_path: Path, chunks_file_path: Path, chunk_len: int):
    assert exists_and_is_file(tokens_file_path)
    assert not chunks_file_path.is_dir()
    # assert chunk_len > 0 and chunk_len == 512
    init_logging()

    with log(f'Processing {tokens_file_path.name}'):
        with log('Calculating total number of chunks'):
            total_chunks = 0
            with read_jsonl_file(tokens_file_path) as tokens_reader:
                for i, line in tqdm(enumerate(tokens_reader)):
                    div, mod = divmod(len(line['tokens']), chunk_len)
                    if (div != 0) and (div != 1):
                        bp()
                    assert div == 0 or div == 1
                    total_chunks += (div if mod == 0 else div + 1)

        with log('Creating chunks and chunks map'):
            _create_chunks_and_map(tokens_file_path, chunks_file_path, chunk_len, total_chunks)


def tokens_files_to_chunks_files(tokens_dir_path: Path, chunks_dir_path: Path, chunk_len: int):
    assert exists_and_is_dir(tokens_dir_path) and exists_and_is_dir(chunks_dir_path)
    # assert chunk_len > 0 and chunk_len == 512

    # swj change
    tokens_file_paths = sorted(list(tokens_dir_path.glob(LZ4_GLOB)))
    # bp()
    # tokens_file_paths = sorted(list(tokens_dir_path.glob("jsonl")))
    print("tokens_file_paths: ", tokens_file_paths)
    with log(f'Processing {len(tokens_file_paths)} tokens files'):
        # for debugging:
        # _tokens_file_to_chunks_files(tokens_file_paths[0], chunks_dir_path / tokens_file_paths[0].name, chunk_len)
        _parallel()(delayed(_tokens_file_to_chunks_files)(tokens_file_path,
                                                          Path(str(chunks_dir_path / tokens_file_path.name) + NPY_SUFFIX),
                                                          chunk_len)
                    for tokens_file_path in track(tokens_file_paths))


def _tokenize_doc(doc: str, tokenizer_cfg: DictConfig):
    assert tokenizer_cfg

    tokenized_doc_outer = tokenize(doc, tokenizer_cfg.name, tokenizer_cfg.repo_or_dir,
                                   tokenizer_cfg.source, tokenizer_cfg.skip_validation, add_special_tokens=True,)
    # get the tokens excluding the CLS and SEP that were added (== the original doc)
    tokenized_doc = tokenized_doc_outer[0, 1:-1]
    tokenized_doc[(tokenized_doc == CLS_TOKEN) | (tokenized_doc == SEP_TOKEN) |  # noqa: W504
                  (tokenized_doc == PAD_TOKEN) | (tokenized_doc == MASK_TOKEN)] = UNK_TOKEN

    return tokenized_doc_outer.tolist()[0]


# TODO: this can be slow if the input files to jsonl_dir_to_docs_text_list_file were large (because only ||izes at file level)
#       could think about splitting files or doing something more sophisticated, if this becomes a problem
def _docs_files_to_tokens_file(docs_file_path, tokens_file_path: Path, tokenizer_cfg: DictConfig):
    assert exists_and_is_file(docs_file_path)
    assert not tokens_file_path.is_dir()
    assert tokenizer_cfg
    init_logging()

    with log(f'Tokenizing doc {docs_file_path.name}'):
        with read_jsonl_file_no_compress(docs_file_path) as docs_reader, write_jsonl_file(tokens_file_path) as tokens_writer:
            for doc_index, doc in tqdm(enumerate(docs_reader)):
                tokens = _tokenize_doc(doc['text'], tokenizer_cfg)[:MAX_TOKENS]
                # swj change
                tokens_writer.write({'tokens': tokens})


def docs_files_to_tokens_files(docs_dir_path: Path, tokens_dir_path: Path, tokenizer_cfg: DictConfig):
    assert docs_dir_path.exists() and docs_dir_path.is_dir()
    assert tokens_dir_path.exists() and tokens_dir_path.is_dir()
    assert tokenizer_cfg is not None

    # swj
    docs_file_paths = sorted(list(docs_dir_path.glob("*.jsonl")))
    # docs_file_paths = sorted(list(docs_dir_path.glob(LZ4_GLOB)))

    # bp()
    with log(f'Tokenizing {len(docs_file_paths)} documents'):
        # debugging: _docs_files_to_tokens_file(docs_file_paths[0], tokens_dir_path / docs_file_paths[0].name, tokenizer_cfg)
        # _docs_files_to_tokens_file(docs_file_paths[0], tokens_dir_path / docs_file_paths[0].name, tokenizer_cfg)
        _parallel()(delayed(_docs_files_to_tokens_file)(docs_file_path, tokens_dir_path / docs_file_path.name, tokenizer_cfg) for docs_file_path in track(docs_file_paths))


def _copy_and_compress(source_file_path: Path, target_file_path: Path):
    assert source_file_path.exists() and source_file_path.is_file()
    assert not target_file_path.is_dir()
    doc_id = 0
    with log(f'Processing file {source_file_path.name}'):
        with jsonlines.open(source_file_path, 'r') as source_reader, write_jsonl_file(target_file_path) as target_writer:
            for line in tqdm(source_reader):
                line["doc_id"] = doc_id
                doc_id += 1
                target_writer.write(line)

# swj
def _copy_and_compress_new(source_file_path: Path, target_file_path: Path, doc_id: int):
    assert source_file_path.exists() and source_file_path.is_file()
    assert not target_file_path.is_dir()
    with log(f'Processing file {source_file_path.name}'):
        with read_jsonl_file_no_compress(source_file_path) as source_reader, write_jsonl_file(target_file_path) as target_writer:
            for line in tqdm(source_reader):
                if 'content' in line:
                    line['text'] = line['content']
                    line.pop('content')
                if len(line["text"].split()[:30]) < 5:  
                    # bp()
                    print("too short, skip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue
                line["doc_id"] = doc_id
                doc_id += 1
                target_writer.write(line)
    return doc_id

def import_docs(source_path: Path, docs_dir_path: Path, globs):
    assert exists_and_is_dir(source_path) and exists_and_is_dir(docs_dir_path)
    assert globs

    docs_file_paths_unsorted = []
    globs_list = globs.split(GLOB_SEPARATOR)
    for glob in globs_list:
        docs_file_paths_unsorted.extend(list(source_path.glob(glob)))

    docs_file_paths = sorted(docs_file_paths_unsorted)
    with log(f'Processing {len(docs_file_paths)} files'):
        # debugging: _copy_and_compress(docs_file_paths[0], Path(str(docs_dir_path / docs_file_paths[0].name) + LZ4_SUFFIX))
        # swj: no multiprocess
        doc_id = 0
        for d in tqdm(docs_file_paths):
            # _copy_and_compress(d, Path(str(docs_dir_path / d.name) + LZ4_SUFFIX))
            doc_id = _copy_and_compress_new(d, Path(str(docs_dir_path / d.name) + LZ4_SUFFIX), doc_id)
            print(f"finish one doc: {d} with doc_id: {doc_id}")
        # _copy_and_compress(docs_file_paths[0], Path(str(docs_dir_path / docs_file_paths[0].name) + LZ4_SUFFIX))
        # _parallel()(delayed(_copy_and_compress)(docs_file_path, Path(str(docs_dir_path / docs_file_path.name) + LZ4_SUFFIX))
                    # for docs_file_path in track(docs_file_paths))
