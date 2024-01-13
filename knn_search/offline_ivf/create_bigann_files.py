
import faiss
import numpy as np
from faiss.contrib.datasets import DatasetBigANN
import argparse
#this will only work on the fair cluster
def main(args: argparse.Namespace):

    data=DatasetBigANN()
    batch_size = 50_000_000
    xb=data.database_iterator(bs=batch_size)
    for i,data_batch in enumerate(xb):
        filename=f"/checkpoint/marialomeli/big_ann_data/bigann{(i):010}.npy"
        print(f"Processing file {i} out of {1_000_000_000_000/batch_size}")
        np.save(filename,data_batch)
        print(f"File {filename} is saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noargs",
        dest="noargs",
        type=str,
        default=True,
        help="Config file",
    )
    args = parser.parse_args()
    main(args)
