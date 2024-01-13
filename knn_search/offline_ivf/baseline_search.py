import argparse
import faiss
import numpy as np
from utils import read_embeddings
import time

'''
Previous benchmmarking:
incontext pretraining previous benchmarking
#load query embeddings, in-context pretraining example
root="/fsx-instruct-opt/swj0419/llama_data/embed/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco/"
file_names = [f"github_oss.chunk.{i:02}.jsonl.lz4.npy" for i in range(32)]
file_names = ["github_oss.chunk.00.jsonl.lz4.npy"]
k=50
'''

def main(args: argparse.Namespace) -> None:

    index_path=args.index_file
    index=faiss.read_index(index_path,faiss.IO_FLAG_ONDISK_SAME_DIR)
    index_ivf=faiss.extract_index_ivf(index)
    index_ivf.nprobe = args.nprobe
    index_ivf.by_residual=True
    res = faiss.StandardGpuResources()
    co = faiss.GpuMultipleClonerOptions()
    co.shard = False #produces index replicas
    co.useFloat16 = True #try with both settings
    #co.common_ivf_quantizer = False #try both settings to see if there's a difference, default is False that each process does the quantiser.
    co.verbose = True
    gpu_index_ivf = faiss.index_cpu_to_all_gpus(index_ivf,co,ngpu=faiss.get_num_gpus())
    k=args.k
    file_name=f"{args.files_common_prefix}_{(args.file_rank):0{args.padding}}.npy"
    fp=args.root+file_name
    _,vecs = read_embeddings(fp,args.dim,dt=np.dtype(np.uint8))
    start = time.time()
    print(f"Start non-batched search with nprobe {args.nprobe} with file {fp}!")
    D, I = gpu_index_ivf.search(vecs, k)
    print("Finished non-batched search!")
    end = time.time()
    total_time = end - start
    print("\n"+ str(total_time))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument(
        "--index_file",
        dest="index_file",
        type=str,
        default="/fsx-ralm-news/marialomeli/offline_faiss/ssnpp/ssnpp_1B/IVF8192_PQ32.faissindex",
        help="Index file path",
    )

    parser.add_argument(
        "--k",
        dest="k",
        type=int,
        default=50,
        help="Number of nearest neighbours",
    )
    parser.add_argument(
        "--root",
        dest="root",
        type=str,
        default="/fsx-ralm-news/marialomeli/ssnpp_data/",
        help="Root for embeddings data",
    )
    parser.add_argument(
        "--files_common_prefix",
        dest="files_common_prefix",
        type=str,
        default="ssnpp",
        help="Embeddings files_common_prefix",
    )
    parser.add_argument(
        "--file_rank",
        dest="file_rank",
        type=int,
        default=1,
        help="number of embedding files",
    )
    parser.add_argument(
        "--dim",
        dest="dim",
        type=int,
        default=256,
        help="Vector dimensionality",
    )
    parser.add_argument(
        "--nprobe",
        dest="nprobe",
        type=int,
        default=128,
        help="number of embedding files",
    )
    parser.add_argument(
        "--padding",
        dest="padding",
        type=int,
        default=10,
        help="number of embedding files",
    )

    args = parser.parse_known_args()[0]
    main(args)
