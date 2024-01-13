
# Offline IVF

This folder contains the code for the offline ivf algorithm that enables to conduct offline big batch search.
![oivf_new](https://github.com/fairinternal/faiss_improvements/assets/4427136/ff96b614-66dd-42cc-968d-4864bf081be6)


Create a fresh conda env:

`conda create --name oivf python=3.10`

`conda activate oivf`

`conda install -c "pytorch/label/nightly" faiss-gpu -c nvidia`

`conda install tqdm`

`conda install pyyaml`

`conda install -c conda-forge submitit`


## Run book

1. Include the new dataset to the yaml file `config_retro.yaml`. You can use `generate_config.py` by specifying the root directory of your dataset and the files with the data shards , e.g. line 49 for the 2T dataset

`python generate_config`

2. Run the train index command
 
`python run.py --command train_index --config config_retro.yaml --xb retro_8B`


3. Run the index-shard command so it computes the indices per shard

`python run.py --command index_shard --config config_retro.yaml --xb retro_8B`


6. Run the search distributed job:
 
`python run.py  --command search --config config_retro.yaml  --xb retro_8B  --run_type cluster --partition learnlab`


Remarks about the `search` command:
a. If the query vectors are different than the database vectors,e.g. retro_100M, it should be passed in the xq argument
b. A new dataset needs to be prepared (following steps 1-3) before passing it to the query vectors argument `–xq`

`python run.py --command search --config config_retro.yaml --xb retro_8B --xq edouard_val`


6. We can always run the consistency-check for sanity checks!

`python run.py  --command consistency_check --config config_retro.yaml --xb retro_8B`

## FAQ

What files are created in the results folder  /checkpoint/gsz/offline_faiss/retro/?

`shard` files: indexes with the vectors per chunk added


## Code formatting

In order to ensure consistent code formatting, use Black to format all code with the line-length set to 120.
