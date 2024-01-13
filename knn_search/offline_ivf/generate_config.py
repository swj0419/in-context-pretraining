import numpy as np
import os
import glob
import yaml
import re
import string


def natural_sort_key(s, _nsre=re.compile("([0-9]+)")):
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


"""
# retro 4B vecs from 10 shards, some merged, some not

root = '/checkpoint/hcir/data/retro-z/repo/dataset_name=1t/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased'

shard_files = [
    [f'attic/train00.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'attic/train01.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'train02.jsonl.lz4.npy'],
    [f'attic/train03.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'attic/train04.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'train05.jsonl.lz4.npy'],
    [f'attic/train06.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'attic/train07.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'attic/train08.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
    [f'attic/train09.jsonl.lz4.npy_{i}_256.npy' for i in range(256)],
]

file_names = [file for shard in shard_files for file in shard]
"""

"""
# retro 4B vecs merged data shards

root = '/checkpoint/hcir/data/retro-z/repo/dataset_name=1t/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased'

file_names = [f'train{i:02}.jsonl.lz4.npy' for i in range(10)]
"""

"""
# retro 100M

root = '/checkpoint/hcir/data/retro-z/repo/dataset_name=30g_redux/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased'

file_names = sorted(glob.glob('*.npy', root_dir=root), key=natural_sort_key)
"""
"""
# retro 2T
root = '/checkpoint/hcir/data/retro-z/repo'

shard_files = [
    [f'dataset_name=2t/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased/train{j}.jsonl.lz4.npy_{i}_1024.npy' for i in range(1024)] for j in range(11, 20)
]

file_names = [
    f'dataset_name=1t/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased/train{j:02}.jsonl.lz4.npy' for j in range(11)
] + [file for shard in shard_files for file in shard]
"""
"""
# edouard_val
root = "/checkpoint/hcir/data/retro-z/repo"

file_names = [
    f"dataset_name=edouard_val/tokenizer=bert-base-cased/seq_len=2048/chunk_len=64/model=bert-base-cased/val.jsonl.lz4.npy"
]
"""
"""
# test_data
root = "/checkpoint/marialomeli/offline_faiss/test_data"
d = 768
dt = np.dtype(np.float32)
file_names = [f"my_data{j:02}.npy" for j in range(2)]

# seamless ENG0,...,ENG5
root = "/fsx-nllb-big/schwenk/mini-mine5/embed.22h1"
file_names = [f"mm5_p5.encf.0{i:02}.eng5" for i in range(55)]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# seamless GLG
root = "/fsx-nllb-big/schwenk/mini-mine5/embed.22h1"
file_names = [f"mm5_p5.encf.0{i:02}.glg" for i in range(6)]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# 1t-swj contriever
root = '/checkpoint/swj0419/dataset_name-1t-swj/tokenizer-facebook_contriever/seq_len-5100/chunk_len-510/model-facebook_contriever'
file_names = [f'train{j:02}.jsonl.lz4.npy' for j in range(11)]
d = 768
dt = np.dtype(np.float32)
"""
"""
root = "/checkpoint/hcir/data/retro-z/repo/dataset_name-1tb/tokenizer-facebook_contriever/chunk_len-64/model-facebook_contriever"
file_names = sorted(glob.glob("*.npy", root_dir=root), key=natural_sort_key)
d = 768
dt = np.dtype(np.float32)
"""
"""
#SONAR ENG1 embeddings
root = "/fsx-nllb-big/kevinheffernan/embeddings-for-maria"
aux = ["008", "009", "010", "011", "018", "019", "029", "032", "050", "051", "052", "062"]
file_names = ["encf.{i}.eng1" for i in aux]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# contriever for REPLUG
root = "/checkpoint/hcir/data/retro-z/repo/dataset_name-1tb_no_fix/tokenizer-contriever/chunk_len-128/model-contriever/"
file_names = [f"train{j:02}.jsonl.lz4.npy" for j in range(11)]
d = 768
dt = np.dtype(np.float32)
"""
"""
# SONAR ENG0 embeddings (first file only)
root = "/checkpoint/kevinheffernan/SONAR/denoise_autoencode_reg2_100k/embed.23h1/eng0"
file_names = ["encf.000.eng0"]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# SONAR ENG0 embeddings (first file only)
root = "/checkpoint/kevinheffernan/SONAR/denoise_autoencode_reg2_100k/embed.23h1/eng0"
file_names = ["encf.000.eng0"]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# SONAR ENG0 embeddings (first file only)
root = "/checkpoint/kevinheffernan/SONAR/denoise_autoencode_reg2_100k/embed.23h1/eng0"
file_names = ["encf.000.eng0"]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# SONAR GLG embeddings
root = "/checkpoint/kevinheffernan/SONAR/denoise_autoencode_reg2_100k/embed.23h1/glg"
file_names = [f"encf.{j:03}.glg" for j in range(7)]
d = 1024
dt = np.dtype(np.float16)
"""
"""
root = "/checkpoint/hcir/data/retro-z/repo/dataset_name-1tb_no_fix/tokenizer-contriever/chunk_len-128/model-contriever"
file_names = [f"train{j:02}.jsonl.lz4.npy_{ell}_256.npy" for ell in range(256) for j in range(1, 10)]
file_names = file_names + ["train00.jsonl.lz4.npy"]
d = 768
dt = np.dtype(np.float32)
"""
"""
# SONAR ENG0 embeddings
root = "/checkpoint/gsz/seamless/sonar/eng0"
file_names = [f"encf.0{i:02}.eng0" for i in range(75)]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# SONAR Chinese speech embeddings
root="/large_experiments/seamless/nllb/data2023H1/embed_m2c2_vad_w2vbert_sonar/cmn/mining_speech_encoder"
file_names = [f"emb_sonar.00{i:03}.cmn" for i in range(322)]
d = 1024
dt = np.dtype(np.float16)
"""
"""
# Sonar Eng speech embeddings
root = "/large_experiments/seamless/nllb/data2023H1/embed_m2c2_vad_w2vbert_sonar/eng_best/mining_speech_encoder"
file_names = [f"emb_sonar.{i:05}.eng" for i in range(30530)]  # check if this are split in eng0,...,eng6
d = 1024
dt = np.dtype(np.float16)
"""
"""
# sonar text embeddings turkish
root = "/fsx-ust/marialomeli/data/seamless/text/tur"
file_names = [f"encf.{i:03}.tur" for i in range(49)]  # check if this are split in eng0,...,eng6
d = 1024
dt = np.dtype(np.float16)
"""
"""
#sonar text embeddings hindi
root ="/fsx-ust/marialomeli/data/seamless/text/hin"
file_names = [f"encf.{i:03}.hin" for i in range(8)]  # check if this are split in eng0,...,eng6
d = 1024
dt = np.dtype(np.float16)
"""
"""
#sonar text embeddings finnish
root ="/fsx-ust/marialomeli/data/seamless/text/fin"
file_names = [f"encf.{i:03}.fin" for i in range(4)]  # check if this are split in eng0,...,eng6
d = 1024
dt = np.dtype(np.float16)
"""
"""
# sonar text embeddings slovak
root = "/fsx-ust/marialomeli/data/seamless/text/slk"
file_names = [f"encf.{i:03}.slk" for i in range(13)]  # check if this are split in eng0,...,eng6
d = 1024
dt = np.dtype(np.float16)
"""
"""
# dragon embeddings ccnet.chunk.00aa.c100.sec0.npy,  ccnet.chunk.01aa.c100.sec42.npy
root = "/checkpoint/victorialin/offline_faiss/dragon_plus_cc350m_embeddings_chunked/all_npy"
aux1 = [f"ccnet.chunk.00a{letter}" for letter in string.ascii_lowercase[:21] for _ in (0, 1)]
aux2 = [f"ccnet.chunk.01a{letter}" for letter in string.ascii_lowercase[:21] for _ in (0, 1)]
first_names = [f"{prefix}.c100.sec{i}.npy" for i, prefix in zip(range(42), aux1)]
second_names = [f"{prefix}.c100.sec{i}.npy" for i, prefix in zip(range(42, 84), aux2)]
file_names = first_names + second_names
d = 768
dt = np.dtype(np.float32)
"""
"""
# trA speech sonar embeddings
root = "/fsx-ust/marialomeli/data/seamless/speech/uzb"
file_names = [f"encf.{i:03}.uzb" for i in range(2)]
d = 1024
dt = np.dtype(np.float16)
# contriever arxiv,b3g,wikipedia,github_oss,stack
root="/fsx-instruct-opt/swj0419/llama_data/embed/arxiv/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root = "/fsx-instruct-opt/swj0419/llama_data/embed/b3g/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root = "/fsx-instruct-opt/swj0419/llama_data/embed/stack/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root = "/fsx-instruct-opt/swj0419/llama_data/embed/github_oss/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root = "/fsx-instruct-opt/swj0419/llama_data/embed/c4/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root= "/fsx-instruct-opt/swj0419/llama_data/embed/github_oss/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root= "/fsx-instruct-opt/swj0419/llama_data/embed/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root = "/fsx-scaling-megabyte/shared/ccnet/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
root= "/fsx-instruct-opt/swj0419/llama_data/embed/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco"
file_names = [f"github_oss.chunk.{i:02}.jsonl.lz4.npy" for i in range(32)]
d = 768
dt = np.dtype(np.float32)

# 1B bigann data
root= "/checkpoint/marialomeli/big_ann_data"
file_names = [f"bigann{i:010}.npy" for i in range(20)]
d = 128
dt = np.dtype(np.float32)
"""
# # ssnpp data
# root= "/fsx-ralm-news/marialomeli/ssnpp_data"
# file_names = [f"ssnpp_{i:010}.npy" for i in range(20)]
# d = 256
# dt = np.dtype(np.uint8)

d = 768
root = "../../output/embed/b3g_large/tokenizer-facebook_contriever_msmarco/seq_len-5100/chunk_len-510/model-facebook_contriever_msmarco/"
file_names = os.listdir(root)
dt = np.dtype(np.float32)



def read_embeddings(fp):
    fl = os.path.getsize(fp)
    nb = fl // d // dt.itemsize
    # print(nb)
    if fl == d * dt.itemsize * nb:  # no header
        return ("raw", np.memmap(fp, shape=(nb, d), dtype=dt, mode="r"))
    else:  # assume npy
        vecs = np.load(fp, mmap_mode="r")
        assert vecs.shape[1] == d
        assert vecs.dtype == dt
        return ("npy", vecs)


cfg = {}
files = []
size = 0
for fn in file_names:
    fp = f"{root}/{fn}"
    # print(fp)
    assert os.path.exists(fp), f"{fp} is missing"
    ft, xb = read_embeddings(fp)
    #print(f"file{fp},file_size:{xb.shape[0]}")
    files.append({"name": fn, "size": xb.shape[0], "dtype": dt.name, "format": ft})
    size += xb.shape[0]

cfg["size"] = size
cfg["root"] = root
# cfg["d"] = d
cfg["files"] = files
print(yaml.dump(cfg))

