import sys

from omegaconf import OmegaConf, DictConfig
from pathlib import Path, PosixPath
import logging
import pprint
import hydra
# import faiss
import os
from ipdb import set_trace as bp
from retro_z_utils import init_logging, log

from retro_z_data_xforms import (
    chunks_and_knns_to_tfds_dataset_parallel,
    chunks_and_knns_to_tfds_dataset_serial,
    parallel_chunks_files_to_embeds_files,
    embeds_dir_plus_index_to_cached_knns,
    tokens_files_to_chunks_files,
    chunks_files_to_embeds_files,
    docs_files_to_tokens_files,
    embeds_dir_to_index,
    embeds_dir_to_index_swj,
    import_docs,
)

'''
['', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages/_pdbpp_path_hack', '/private/home/hcir/.conda/envs/retro/lib/python38.zip', '/private/home/hcir/.conda/envs/retro/lib/python3.8', '/private/home/hcir/.conda/envs/retro/lib/python3.8/lib-dynload', '/private/home/swj0419/.local/lib/python3.8/site-packages', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages', '/private/home/hcir/src2/richjames0/autofaiss', '/private/home/hcir/src/richjames0/RETRO-pytorch', '/private/home/hcir/src/sksq96/pytorch-summary', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', "/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages"]
'''
# TODO: pull these out
import sys
# sys.path.append("/private/home/swj0419/rlm_pretrain/hcir/retro-z/retro_z/RETRO-pytorch")
from retro_pytorch.retrieval import get_tokenizer, get_bert

# BookCorpusFair_10.jsonl.lz4.npy
TFDS_DATASET_VERSION = '1.0.0'
# 2x number of CPUs on devfair
FAISS_NUM_THREADS = 160


def get_source_dir_path(cfg: DictConfig):
    return Path(cfg.documents.source_path)


def get_documents_dir_path(cfg: DictConfig):
    return Path(cfg.base_path) / f'{str(cfg.dataset_name)}'


def get_normalized_tokenizer_name(cfg: DictConfig):
    # / in tokenizer name is not valid for a directory name
    return cfg.tokens.tokenizer.name.replace('/', '_').replace('-', '_')


def get_tokens_dir_path(cfg: DictConfig):
    return get_documents_dir_path(cfg) / f'tokenizer-{get_normalized_tokenizer_name(cfg)}'


def get_sequences_dir_path(cfg: DictConfig):
    return get_tokens_dir_path(cfg) / f'seq_len-{cfg.sequences.seq_len}'


def get_chunks_dir_path(cfg: DictConfig):
    return get_sequences_dir_path(cfg) / f'chunk_len-{cfg.chunks.chunk_len}'


def get_normalized_model_name(cfg: DictConfig):
    # / in model name is not valid for a directory name
    return cfg.embeddings.model.name.replace('/', '_').replace('-', '_')


def get_embeddings_dir_path(cfg: DictConfig):
    return get_chunks_dir_path(cfg) / f'model-{get_normalized_model_name(cfg)}'


def get_reference_index_dir_path(cfg: DictConfig):
    index_reference_as_dir_name = cfg.index.index_reference.replace('/', '.')
    return get_embeddings_dir_path(cfg) / f'index_reference-{index_reference_as_dir_name}'


def get_index_dir_path(cfg: DictConfig):
    return get_embeddings_dir_path(cfg) / f'index_string-{cfg.index.index_string}'


def get_knns_dir_path(cfg: DictConfig):
    # TODO: consider integrating get_index and get_reference_index and moving this logic into that function (see also below)
    if cfg.index.index_reference:
        return get_reference_index_dir_path(cfg) / f'k-{cfg.precalculated_knns.k}'
    else:
        return get_index_dir_path(cfg) / f'k-{cfg.precalculated_knns.k}'


def get_tfds_dir_path(cfg):
    # TODO: as above, regarding factoring this stuff better
    # TODO: this is duplcated above
    # we don't need the full index reference since would be unwieldy and params should match, so we'll just take dataset name
    index_string = f"ref_{cfg.index.index_reference.split('/')[0].split('-')[1]}" if cfg.index.index_reference else cfg.index.index_string

    dataset_name = f'{cfg.dataset_name}_{get_normalized_tokenizer_name(cfg)}_{cfg.sequences.seq_len}_{cfg.chunks.chunk_len}_' \
                   f'{get_normalized_model_name(cfg)}_{index_string}_{cfg.precalculated_knns.k}_{cfg.tfds_dataset.k}'

    return get_knns_dir_path(cfg) / f'k-{cfg.tfds_dataset.k}' / dataset_name / TFDS_DATASET_VERSION

'''
index: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat')
knn_dir_path: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50')
embeddings_file_path: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/train01.jsonl.lz4.npy')
knns: query_id2docid: /checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50/knns.npy
knns_dists: query_id2docid: /checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50/knns.npy.dist

'''

def generate_precalculated_knns(cfg: DictConfig):
    index_reference = cfg.index.index_reference

    with log(f'Generating precalcuated neighbors{" to reference index" if index_reference is not None else ""}'):
        # TODO: (see also in knns, above) consider merging get_index_dir_path and having it support both these modes conditionally
        if index_reference:
            index_dir_path = Path(cfg.base_path) / cfg.index.index_reference
        else:
            index_dir_path = get_index_dir_path(cfg)
        # bp()
        knns_dir_path = get_knns_dir_path(cfg)
        os.makedirs(knns_dir_path, exist_ok=True)
        batch_size = cfg.precalculated_knns.batch_size
        embed_dim = cfg.embeddings.embed_dim
        k = cfg.precalculated_knns.k
        enable_gpu = cfg.precalculated_knns.enable_gpu

        embeds_dir_plus_index_to_cached_knns(index_reference, index_dir_path, get_embeddings_dir_path(cfg),
                                             knns_dir_path, batch_size, embed_dim, k, enable_gpu)


def generate_reference_index(cfg: DictConfig):
    with log('Generating reference index'):
        reference_index_dir_path = get_reference_index_dir_path(cfg)
        os.makedirs(reference_index_dir_path, exist_ok=True)


def generate_index(cfg: DictConfig):
    with log('Generating index'):
        index_dir_path = get_index_dir_path(cfg)
        os.makedirs(index_dir_path, exist_ok=True)
        # bp()
        if cfg.index.index_string == 'flat':
            embeds_dir_to_index(get_embeddings_dir_path(cfg), index_dir_path, cfg.index.index_string, cfg.embeddings.embed_dim)
        elif cfg.index.index_string == 'ivfpq':
            embeds_dir_to_index_swj(get_embeddings_dir_path(cfg), index_dir_path, cfg.index.index_string, cfg.embeddings.embed_dim)


def generate_embeddings(cfg: DictConfig):
    with log('Generating embeddings'):
        embeddings_dir_path = get_embeddings_dir_path(cfg)
        os.makedirs(embeddings_dir_path, exist_ok=True)
        print("get_chunks_dir_path(cfg): ", get_chunks_dir_path(cfg))
        # # # swj: to be deleted
        # chunks_files_to_embeds_files(get_chunks_dir_path(cfg), embeddings_dir_path, cfg.embeddings.model,
        #                                  cfg.chunks.chunk_len, cfg.embeddings.batch_size, cfg.embeddings.embed_dim)
        # 1/0
        # bp()
        if cfg.embeddings.parallel.num_workers == 1:
            chunks_files_to_embeds_files(get_chunks_dir_path(cfg), embeddings_dir_path, cfg.embeddings.model,
                                         cfg.chunks.chunk_len, cfg.embeddings.batch_size, cfg.embeddings.embed_dim)
        else:
            # bp()
            print("parallel")
            parallel_chunks_files_to_embeds_files(get_chunks_dir_path(cfg), embeddings_dir_path, cfg.embeddings.batch_size,
                                                  cfg.embeddings.model, cfg.embeddings.embed_dim, cfg.chunks.chunk_len,
                                                  cfg.embeddings.parallel)


def generate_chunks(cfg: DictConfig):
    with log('Generating chunks'):
        chunks_dir_path = get_chunks_dir_path(cfg)
        os.makedirs(chunks_dir_path, exist_ok=True)
        tokens_files_to_chunks_files(get_tokens_dir_path(cfg), chunks_dir_path, cfg.chunks.chunk_len)


def generate_tokens(cfg: DictConfig):
    with log('Generating tokens'):
        tokens_dir_path = get_tokens_dir_path(cfg)
        os.makedirs(tokens_dir_path, exist_ok=True)
        tokens_dir_path = tokens_dir_path
        # swj
        # print(cfg.source_path)
        # bp()
        # docs_files_to_tokens_files(get_documents_dir_path(cfg), tokens_dir_path, cfg.tokens.tokenizer)
        docs_files_to_tokens_files(PosixPath(cfg.source_path), tokens_dir_path, cfg.tokens.tokenizer)


def generate_documents(cfg: DictConfig):
    with log('Generating documents'):
        source_path = get_source_dir_path(cfg)
        # bp()
        globs = cfg.documents.glob
        docs_dir_path = get_documents_dir_path(cfg)
        os.makedirs(docs_dir_path, exist_ok=True)
        import_docs(source_path, docs_dir_path, globs)


def generate_tfds_dataset(cfg: DictConfig):
    tfds_dir_path = get_tfds_dir_path(cfg)
    os.makedirs(tfds_dir_path, exist_ok=True)

    split = 'train'
    embeds_dir_path_modeling = get_embeddings_dir_path(cfg)
    chunks_dir_path_modeling = get_chunks_dir_path(cfg)
    if cfg.index.index_reference:
        split = 'validation'
        # for validation, we do language modeling on the validation set, but retrieve from the training set
        # TODO: make this less hacky + all kinds of validation we could do here including tokenizers etc.
        embeds_dir_path_retrieval = Path(cfg.base_path) / Path(cfg.index.index_reference).parent
        chunks_dir_path_retrieval = embeds_dir_path_retrieval.parent
    else:
        # for training we retrieve from the same set as we do language modeling
        embeds_dir_path_retrieval = embeds_dir_path_modeling
        chunks_dir_path_retrieval = chunks_dir_path_modeling
    # bp()
    if cfg.tfds_dataset.parallel.distribute:
        '''
        chunks_and_knns_to_tfds_dataset_parallel(tfds_dir_path: Path, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                             embeds_dir_path_modeling, embeds_dir_path_retrieval,
                                             knns_dir_path: Path, chunk_len: int, seq_len: int, K: int, k_: int,  # noqa: N803
                                             parallel_cfg: DictConfig, split: str, log_every_num_chunks: int)
        '''
        chunks_and_knns_to_tfds_dataset_parallel(tfds_dir_path, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                                 embeds_dir_path_modeling, embeds_dir_path_retrieval,
                                                 get_knns_dir_path(cfg), cfg.chunks.chunk_len, cfg.sequences.seq_len,
                                                 cfg.precalculated_knns.k, cfg.tfds_dataset.k, cfg.tfds_dataset.parallel,
                                                 split, cfg.tfds_dataset.log_to_jsonl_every)
    else:
        pass
        # TODO: Implement or remove serial option
        '''
         chunks_and_knns_to_tfds_dataset_serial(tfds_dir_path: Path, chunks_dir_path: Path, embeddings_dir_path: Path,
                                           knns_dir_path: Path, chunk_len: int, seq_len: int, K: int, k_: int, split: str):  # noqa: N803
        '''
        chunks_and_knns_to_tfds_dataset_serial(tfds_dir_path, chunks_dir_path_modeling, chunks_dir_path_retrieval,
                                          get_knns_dir_path(cfg), cfg.chunks.chunk_len, cfg.sequences.seq_len, cfg.precalculated_knns.k, cfg.tfds_dataset.k, split)

        # chunks_and_knns_to_tfds_dataset_serial(tfds_dir_path, chunks_dir_path_modeling, chunks_dir_path_retrieval, get_embeddings_dir_path(cfg),
        #                                        get_knns_dir_path(cfg), cfg.chunks.chunk_len, cfg.sequences.seq_len,
        #                                        cfg.precalculated_knns.k, cfg.tfds_dataset.k, split)


def validate_config_interdependencies(cfg: DictConfig):
    assert (cfg.sequences.seq_len % cfg.chunks.chunk_len) == 0, 'Sequence length must be divisible by chunk size'


@hydra.main(config_path="configs", config_name="retro_z_data_llama", version_base="1.2")
def main(cfg: DictConfig):
    init_logging()
    logging.info(f'Executing with config:\n {pprint.pformat(OmegaConf.to_object(cfg))}')
    logging.critical('Press enter to continue'); input()

    validate_config_interdependencies(cfg)

    if (cfg.index.generate and not cfg.index.index_reference) or cfg.precalculated_knns.generate:
        faiss.omp_set_num_threads(FAISS_NUM_THREADS)

    if cfg.documents.generate:
        generate_documents(cfg)

    if cfg.tokens.generate:
        get_tokenizer()
        generate_tokens(cfg)

    if cfg.chunks.generate:
        generate_chunks(cfg)

    if cfg.embeddings.generate:
        # get_bert()
        generate_embeddings(cfg)

    if cfg.index.generate:
        if not cfg.index.index_reference:
            generate_index(cfg)
        else:
            generate_reference_index(cfg)

    if cfg.precalculated_knns.generate:
        generate_precalculated_knns(cfg)

    if cfg.tfds_dataset.generate:
        generate_tfds_dataset(cfg)


if __name__ == '__main__':
    main()
