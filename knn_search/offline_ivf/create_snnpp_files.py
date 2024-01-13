
import faiss
import numpy as np
from faiss.contrib.datasets import DatasetBigANN
import argparse
from utils import xbin_mmap
DATA_BATCH:int=50000000
NUM_FILES:int=1000000000//DATA_BATCH
#this will only work on the fair cluster

def main(args: argparse.Namespace):

    filepath="/datasets01/big-ann-challenge-data/FB_ssnpp/FB_ssnpp_database.u8bin"
    ssnpp_data=xbin_mmap(fname=filepath,dtype="uint8")

    for i in range(20):
        xb_batch=ssnpp_data[i*DATA_BATCH:(i+1)*DATA_BATCH,:]
        filename=f"/checkpoint/marialomeli/ssnpp_data/ssnpp_{(i):010}.npy"
        np.save(filename,xb_batch)
        print(f"File {filename} is saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--noargs",
        dest="noargs",
        type=str,
        default="no args",
        help="Config file",
    )
    args = parser.parse_args()
    main(args)