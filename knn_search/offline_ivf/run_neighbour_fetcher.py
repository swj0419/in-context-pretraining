import argparse
from utils import load_config, create_folder_and_write_results
from neighbour_vector_fetcher import NeighbourVectorFetcher
import numpy as np
from pathlib import Path
from utils import NUM_QUERIES


def main(args: argparse.Namespace):
    """
    Load the approximate search matrix results of nearest neighbour indices for each query,
    fetch the vectors corresponding to these and compute the original distances with the corresponding query vector.

    """
    config = load_config(args.config)

    nvf = NeighbourVectorFetcher(
        queries_neighbour_indices_path=args.queries_neighbour_indices,
        config=args.config,
        xb=args.xb,
        xq=args.xq,
        num_queries=args.num_queries,
    )
    D_good = nvf.compute_neighbours_distances()
    filename_specs = Path(args.queries_neighbour_indices).stem
    if "approx" in filename_specs:
        filename_specs = filename_specs.split(
            "approx",
        )[1]

    create_folder_and_write_results(
        output_path=config["output"] + "/groundtruth_eng_0/",
        filename="D_neighbours_xb_" + f"{nvf.xb}" + "_with_xq_" + f"{nvf.xq}" + filename_specs,
        object_to_save=D_good,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        required=True,
        help="Config file",
    )
    parser.add_argument(
        "--queries_neighbour_indices",
        dest="queries_neighbour_indices",
        type=str,
        required=True,
        help="I matrix path",
    )
    parser.add_argument(
        "--xb",
        dest="xb",
        type=str,
        required=True,
        help="Name of dataset for database vectors",
    )

    parser.add_argument(
        "--xq",
        dest="xq",
        type=str,
        default=None,
        help="Name of dataset for query vectors",
    )

    parser.add_argument(
        "--num_queries",
        dest="num_queries",
        type=str,
        default=NUM_QUERIES,
        help="Number of queries for subset",
    )

    args = parser.parse_args()
    main(args)
    # for the ground truth:
    # python run_neighbour_fetcher.py --config config_seamless.yaml --xb eng_0 --queries_neighbour_indices "/checkpoint/marialomeli/offline_faiss/seamless/groundtruth_eng_0/I_16.npy" --xq eng_3
    # for approximate search with nprobe 128:
    # python run_neighbour_fetcher.py --config config_seamless.yaml --xb eng_0 --queries_neighbour_indices "/checkpoint/marialomeli/offline_faiss/seamless/groundtruth_eng_0/I_approx_k_16_nprobe_128.npy" --xq eng_3
