#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_in
#SBATCH --output=/fsx-ralm-news/%u/offline_faiss/logs/%A.out
#SBATCH --error=/fsx-ralm-news/%u/offline_faiss/logs/%A.err
#SBATCH --signal=USR1@140
#SBATCH --open-mode=append
#SBATCH --partition=learnlab


srun python3 baseline_search.py --file_rank 0 --index_file "/fsx-ralm-news/marialomeli/offline_faiss/ssnpp/ssnpp_1B/IVF16384_PQ32.faissindex"