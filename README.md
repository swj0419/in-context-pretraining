
# in-context-pretraining

## Installation

Set up your environment using the following commands:

```bash
conda create -n iclm python=3.10
conda activate iclm
conda install -c "pytorch/label/nightly" faiss-gpu -c nvidia
pip install -r requirements.txt
```

## Sort Pretraining Data using In-Context Pretraining

We provide an example corpus in `data/b3g` to demonstrate our pipeline. The corpus is in jsonl format (`data/b3g/chunk.jsonl`), with each line representing one document. To use your own corpus, place it under `data` and update the data directory in the relevant configuration file.

### Retrieve Neighbor Documents

#### Encode Documents into Embeddings

Tokenize documents and generate embeddings for each document in the corpus:

```bash
cd build_embeds
python retro_z_data.py --config-name example_config
```

- Modify `source_path` in `example_config` to specify the data directory (default is `data/b3g`).
- Output embedding directory is set in `base_path`.
- The program submits jobs using the slurm system. Configure slurm settings in lines 30-39 of `example_config`.


#### Efficient kNN search

1. An example config is shown in `configs/config_test.yaml`. You first need to set FAISS related hyperparameters (like Line 2 to Line 23) and use generate_config.py by specifying the embedding dimension and root directory to generate the rest of the config. 

`python generate_config`

2. Run the train index command
 
`python run.py --command train_index --config configs/config_test.yaml --xb ccnet_new --no_residuals`


3. Run the index-shard command so it computes the indices per shard

`python run.py --command index_shard --config configs/config_test.yaml --xb ccnet_new`


4. Run the search distributed job:
 
`python run.py  --command search --config configs/config_test.yaml --xb ccnet_new  --cluster_run --partition learnlab`


Remarks about the `search` command:
a. If the query vectors are different than the database vectors,e.g. retro_100M, it should be passed in the xq argument
b. A new dataset needs to be prepared (following steps 1-3) before passing it to the query vectors argument `–xq`

`python run.py --command search --config configs/config_test.yaml --xb ccnet_new --xq edouard_val`


5. We can always run the consistency-check for sanity checks!

`python run.py  --command consistency_check --config configs/config_test.yaml --xb ccnet_new`

<!-- 
```
cd knn_search/offline_ivf
python generate_config.py > config_test.yaml
python run.py --command train_index --config config_test.yaml --xb b3g

``` -->

> **Note:** Following this procedure, a `npy` file will be generated, containing the results of the kNN search. Notably, the first row in this file corresponds to the kNN for the first document in the original jsonl dataset.


#### Sort documents based on kNNs
Once you have kNNs, you can sort your entire pretraining documents using the following command. It organizes your documents based on their kNN relationships:
```
# Set your domain and directories
DOMAIN=b3g
TEXT_DIR=data  # The original pretraining corpus directory: the code uses ${TEXT_DIR}/$DOMAIN as the input di
OUTPUT_DIR=[Your output directory]
KNN_DIR=[Directory containing kNNs from previous steps]

# Run the sorting script
python sort.py --domain $DOMAIN --output_dir $OUTPUT_DIR --knn_dir $KNN_DIR --text_dir $TEXT_DIR
```


🚨**Note**🚨: : After completing the sorting process, you will have a final jsonl file. This file is organized in such a way that documents which are closely related, as determined by the kNN results, are grouped together. When your pretraining code reads this file line by line, it will encounter related documents not only within the same context but also between adjacent contexts. However, it is crucial to maintain document similarity only within the same input context and not across adjacent contexts. Your pretraining code might require additional preprocessing to ensure **diversity** between adjacent contexts.

