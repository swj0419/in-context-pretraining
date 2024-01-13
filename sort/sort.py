import sys
from tqdm import tqdm
import argparse
import pickle
from collections import OrderedDict
import statistics
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import numpy as np
import random
from collections import defaultdict
import time
import json

import os
from ipdb import set_trace as bp

# from src.data.tokenizer import Tokenizer

from check_match import *

random.seed(0)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# compare two documents using n-gram similarity
def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams.add(ngram)
    return ngrams


def ngram_similarity(doc1, doc2, n):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)


class sort_class():
    def __init__(self, id2file_pos, output_file, context_len, text_key, text_dir, knn_dir, domain):
        self.num_docs = len(id2file_pos)
        self.seen_docs = set()
        self.unseen_docs = set(range(self.num_docs))
        print(f"num docs: {self.num_docs}")
        self.id2file_pos = id2file_pos
    

        self.doc_sim_threshold = 0.85
        self.n = 3
        self.context_len = context_len
        self.output_file = output_file
        self.text_key = text_key
        self.text_dir = text_dir
        self.knn_dir = knn_dir
        self.file_name_base = "I0000000000"

        # tokenizer_path = "/fsx-onellm/shared/cm3leon_model/bpe_tokenizer/gpt2-unified-image-sentinel.json"
        # self.tokenizer = HFTokenizer(tokenizer_path, 0, 1024, rotation_prob=0.0, retrieval_prob=0.0, num_retrieved_docs=0)
        self.domain = domain

        self.cur_k = None
        self.filter_docs = []
        
        self.cluster2docs = defaultdict(list)
        self.doc2cluster = {}

        self.num_docs_per_file = 50000000

        # self.knns = np.load(f"{self.knn_dir}/{file_name}_IVF32768_PQ256_np64.npy", mmap_mode="r")
        self.all_knns = []
        self.cluster_size = 21

    def load_all_knns(self):
        for i in range(0, self.num_docs, self.num_docs_per_file):
            file_name = self.file_name_base[:-len(str(i))] + str(i)
            knns = np.load(f"{self.knn_dir}/{file_name}_IVF32768_PQ256_np64.npy", mmap_mode="r")
            self.all_knns.append(knns)
            print(f"load knn: {i}")


    def dump_filter_docs(self):
        pickle_dump(self.filter_doc, f"{self.output_file}/filtered_docs.pkl")


    def load_corresponding_knns(self, query_id, num_docs_per_file, file_name_base):
        # start = time.time()
        file_id = (query_id // num_docs_per_file) 
        relative_id = query_id % num_docs_per_file
        # bp()
        # file_id = file_id * num_docs_per_file
        # file_name = file_name_base[:-len(str(file_id))] + str(file_id)
        # print(f"compute knn file: {time.time() - start}")
        # knns = np.load(f"{self.knn_dir}/{file_name}_IVF32768_PQ256_np64.npy", mmap_mode="r")
        knns = self.all_knns[file_id]
        # print(f"load knn time: {time.time() - start}")
        return knns, relative_id

    def sort(self):
        # load knns
        self.load_all_knns()

        # cluster
        cluster_id = 0
        cur_cluster_len = 0

        # first doc
        self.cur_k = self.unseen_docs.pop()
        self.cluster2docs[cluster_id].append(self.cur_k)
        self.seen_docs.add(self.cur_k)
        with tqdm(total=self.num_docs-1) as pbar:
             while self.unseen_docs:
                # start_time = time.time()
                knns, relative_id = self.load_corresponding_knns(self.cur_k, self.num_docs_per_file, self.file_name_base)
                # knns, relative_id = self.all_knns[0], self.cur_k
                # print(f"load knn time: {time.time() - start_time}")

                # start_time = time.time()
                knn = knns[relative_id, :]
                # print(f"get knn time: {time.time() - start_time}")

                # start_time = time.time()
                first_doc = self.output_first_doc_knn(knn)
                # print(f"first doc time: {time.time() - start_time}")

                if (first_doc is None) or (cur_cluster_len >= self.cluster_size):
                    # start_time = time.time()
                    self.cur_k = self.unseen_docs.pop()
                    # print(f"random time: {time.time() - start_time}")
                    cluster_id += 1
                    cur_cluster_len = 0
                else:
                    self.cur_k = first_doc
                    self.unseen_docs.remove(self.cur_k)
                # start_time = time.time()
                self.cluster2docs[cluster_id].append(self.cur_k)
                self.doc2cluster[self.cur_k] = cluster_id
                cur_cluster_len += 1
                self.seen_docs.add(self.cur_k)
                pbar.update(1)
                # print(f"add time: {time.time() - start_time}")
                # bp()
                # assert len(self.seen_docs) + len(self.unseen_docs) == self.num_docs
        pickle_dump(self.cluster2docs, f"{self.output_file}/cluster2docs.pk")
        pickle_dump(self.doc2cluster, f"{self.output_file}/doc2cluster.pk")
    
    def build_doc2_cluster(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            for doc in docs:
                self.doc2cluster[doc] = cluster_id
        pickle_dump(self.doc2cluster, f"{self.output_file}/doc2cluster.pk")

    def build_cluster2length(self):
        length_list = []
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        self.cluster2length = {}
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            self.cluster2length[cluster_id] = len(docs)
            length_list.append(len(docs))
        print(f"average length: {sum(length_list)/len(length_list)}")
        # bp()


    def merge(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        self.doc2cluster = pickle_load(f"{self.output_file}/doc2cluster.pk")
        # data_stats(self.cluster2docs)

        merged_clusters_num = 0
        for cluster, cluster_docs in tqdm(self.cluster2docs.copy().items()):
            if len(cluster_docs) < self.cluster_size:
                merged_clusters_num += 1
                # print(merged_clusters_num)
                for doc in cluster_docs:
                    knns, relative_id = self.knns, doc
                    top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(knns[relative_id, :], cluster)
                    # bp()
                    k_cluster_docs = self.cluster2docs[top1k_cluster]
                    # bp()
                    # add k to doc
                    # k_cluster_docs.append(k)
                    k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)

                    # update the cluster
                    self.cluster2docs[top1k_cluster] = k_cluster_docs
                    self.doc2cluster[doc] = top1k_cluster
                del self.cluster2docs[cluster]
        print(merged_clusters_num)
        pickle_dump(self.cluster2docs, f"{self.output_file}/cluster2docs_merge.pk")
        
    def analyze_data(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs_merge.pk")
        data_stats(self.cluster2docs)
        
        
    def output_first_doc_knn(self, knn):
        for k in knn[1:10]:
            if k not in self.seen_docs:
                return k
        return None


    def output_first_doc_knn_not_in_the_cluster(self, knn, cluster_id):
        for k in knn[1:10]:
            k_cluster = self.doc2cluster[k]
            # bp()
            while k_cluster != cluster_id:
                return k, k_cluster
        return None, None

    def write_docs(self):
        sort_doc = self.cluster2list()
        output_folder = f"{self.output_file}/data"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Determine the number of processes to use
        num_processes = 32
        chunks = self.divide_into_chunks(sort_doc, num_processes)
        
        # Create a pool of workers and distribute the work
        args_list = [] 
        for i, chunk in enumerate(chunks):
            args_list.append((chunk, i))

        print(f"data ready: ", len(args_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap(self.write_docs_wrapper, args_list), total=len(args_list)):
                pass


    def divide_into_chunks(self, lst, n):
        """Divide a list into n chunks."""
        # each bucket size is len(lst)/n
        batch_size = len(lst) // n
        for i in tqdm(range(0, len(lst), batch_size)):
            yield lst[i:i + batch_size]


    def write_docs_wrapper(self, args):
        return self.write_docs_single(*args)

    def write_docs_single(self, sort_doc_chunk, file_index):
        output_folder = f"{self.output_file}/data"
        prev_doc = None
        filter_docs = []
        with open(f"{output_folder}/train_{file_index}.jsonl", "w") as f:
            for doc_id in tqdm(sort_doc_chunk):
                doc = get_document_at_position(self.text_dir, self.id2file_pos, doc_id, self.domain)
                if prev_doc is not None:
                    try:
                        doc_sim = ngram_similarity(doc[self.text_key][:100], prev_doc[self.text_key][:100], self.n)
                    except:
                        print("None doc")
                        print(doc)
                        filter_docs.append(self.cur_k)
                        continue
                    if doc_sim > self.doc_sim_threshold:
                        filter_docs.append(self.cur_k)
                        continue
                f.write(json.dumps(doc) + "\n")
                prev_doc = doc
        print(f"filter docs: {len(filter_docs)}")


    def cluster2list(self):
        self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs_merge.pk")
        sort_doc = []
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            sort_doc.extend(docs)
        return sort_doc


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="wikipedia")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--knn_dir", type=str, default="/fsx-onellm/shared/from_rsc")
    parser.add_argument("--text_dir", type=str, default="/fsx-onellm/data/corpora/text_only")

    args = parser.parse_args()
    domain = args.domain

    # output dir
    output_dir = f"{args.output_dir}/{domain}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load knn
    knn_dir = f"{args.knn_dir}/{domain}/knn"
    file_path = f"{knn_dir}/I0000000000_IVF32768_PQ256_np64.npy"
    # knns = load_knn(file_path)

    text_dir = f"{args.text_dir}/{domain}"
    file_id_text2count = check_doc_line_count(output_dir, text_dir, domain)
    # build id2file_pos
    if os.path.exists(f"{output_dir}/id2file_pos.pkl"):
        id2file_pos = pickle_load(f"{output_dir}/id2file_pos.pkl")
    else:
        id2file_pos = build_index_all_file(text_dir, output_dir, domain, file_id_text2count)

    sort_member = sort_class(id2file_pos, output_dir, 4096, "text", text_dir, knn_dir, domain)

    # sort_member.build_cluster2length()
    sort_member.sort()
    sort_member.merge()
    sort_member.analyze_data()
    sort_member.write_docs()

    






    