import numpy as np
import os
import json
from ipdb import set_trace as bp
from tqdm import tqdm
import pickle
import statistics
import subprocess
from multiprocessing import Pool



def load_knn(file_path):
    knns = np.load(file_path, mmap_mode="r")
    return knns

def build_index(text_file, id2file_pos, file_id_text, doc_id):
    with open(text_file, 'r') as file:
        position = file.tell()
        line = file.readline()
        with tqdm(total=1130243) as pbar:
            while line:
                id2file_pos[doc_id] = [file_id_text, position]
                # bp()
                doc_id += 1
                position = file.tell()
                line = file.readline()
                pbar.update(1)
                # tqdm.update(1)
    return id2file_pos

def get_doc_id(file_id, file_id_text2count):
    doc_id = 0
    for i in range(file_id):
        doc_id += file_id_text2count[i]
    return doc_id


    

# def build_index_all_file(text_dir, output_index_dir, domain, file_id_text2count):
#     id2file_pos = {}
#     for file_id in tqdm(file_id_text2count.keys()):
#         file_id_text = format_number(file_id)
#         text_file = f"{text_dir}/{domain}.chunk.{file_id_text}.jsonl"
#         doc_id = get_doc_id(file_id, file_id_text2count)
#         id2file_pos = build_index(text_file, id2file_pos, file_id_text, doc_id)
#         with open(f"{output_index_dir}/id2file_pos_{file_id}.pkl", "wb") as f:
#             pickle.dump(id2file_pos, f)


def build_index_wrapper(args):
    """
    Wrapper function for build_index to pass multiple arguments.
    """
    return build_index(*args)

def build_index_all_file(text_dir, output_index_dir, domain, file_id_text2count):
    if not os.path.exists(os.path.join(output_index_dir, f"id2file_pos.pkl")):
        args_list = []
        for file_id in file_id_text2count.keys():
            file_id_text = format_number(file_id)
            text_file = os.path.join(text_dir, f"{domain}.chunk.{file_id_text}.jsonl")
            doc_id = get_doc_id(file_id, file_id_text2count)
            args_list.append((text_file, {}, file_id_text, doc_id))

        print(f"Number of files to process: {len(args_list)}")
        # Using multiprocessing Pool
        with Pool(len(file_id_text2count)) as pool:
            results = list(tqdm(pool.imap(build_index_wrapper, args_list), total=len(args_list)))

        # Merging results and dumping
        id2file_pos = {}
        for i, result in enumerate(results):
            id2file_pos.update(result)

        with open(os.path.join(output_index_dir, f"id2file_pos.pkl"), "wb") as f:
            pickle.dump(id2file_pos, f)
    else:
        with open(os.path.join(output_index_dir, f"id2file_pos.pkl"), "rb") as f:
            id2file_pos = pickle.load(f)
    return id2file_pos


def format_number(number):
    """
    Formats a given number. If it's a single digit (less than 10), it prefixes a zero.
    Otherwise, it returns the number as a string.
    """
    if 0 <= number < 10:
        return f'0{number}'
    else:
        return str(number)
    

def check_doc_line_count(output_index_dir, text_dir, domain):
    if os.path.exists(f"{output_index_dir}/file_id_text2count.pkl"):
        file_id_text2count = pickle_load(f"{output_index_dir}/file_id_text2count.pkl")
    else:
        file_id_text2count = {}
        for file_id in tqdm(range(32)):
            file_id_text = format_number(file_id)
            text_file = f"{text_dir}/{domain}.chunk.{file_id_text}.jsonl"
            try:
                # Using wc -l command to count lines
                result = subprocess.run(['wc', '-l', text_file], capture_output=True, text=True)
                if result.returncode == 0:
                    line_count = int(result.stdout.split()[0])
                    file_id_text2count[file_id] = line_count
                    print(f"File {text_file} has {line_count} lines.")
                else:
                    print(f"Error in processing file {text_file}: {result.stderr}")
            except Exception as e:
                print(f"An error occurred while processing {text_file}: {e}")
        pickle_dump(file_id_text2count, f"{output_index_dir}/file_id_text2count.pkl")
    return file_id_text2count


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj

def get_document_at_position(text_dir, id2file_pos, id, domain):
    file_id_text, position = id2file_pos[id]
    jsonl_file_path = f"{text_dir}/{domain}.chunk.{file_id_text}.jsonl"
    with open(jsonl_file_path, 'r') as file:
        # print(position)
        file.seek(position)
        line = file.readline()
        return json.loads(line)

def data_stats(clusterid2docids):
    clusterid2count  = {}
    total_docs = 0
    cluster_21 = 0
    for k, v in tqdm(clusterid2docids.items()):
        clusterid2count[k] = len(v)
        total_docs += len(v)
        if len(v) >= 21:
            cluster_21 += 1
    count_list = list(clusterid2count.values())
    q = statistics.quantiles(count_list, n=100)
    print(f"quantiles 25%: {q[25]}, 50%: {q[50]}, 75%: {q[75]}")
    print(f"total_docs: ", total_docs)
    # number of clusters with more than 21 docs:
    useful_cluster = len([i for i in count_list if i >= 21])
    # bp()
    print(f"number of clusters with more than 21 docs: ", useful_cluster)
