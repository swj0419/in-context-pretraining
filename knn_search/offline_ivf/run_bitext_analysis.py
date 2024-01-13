import pandas as pd

finnish_PQ64 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/baseline/fin/eng0-fin.TH-1.TH-1.06.bitext.TH-1.TH-1.06.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=None,
    sep="\t",
)
finnish_PQ64.columns = ["margin score", "lang1", "lang2"]
print(finnish_PQ64["margin score"].describe())

finnish_PQ512 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/faiss/fin/eng0-fin.TH-1.TH-1.13.bitext.TH-1.TH-1.13.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=None,
    sep="\t",
)
finnish_PQ512.columns = ["margin score", "lang1", "lang2"]
print(finnish_PQ512["margin score"].describe())

print("hindi_PQ64:")
hindi_PQ64 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/baseline/hin/eng0-hin.TH-1.TH-1.09.bitext.TH-1.TH-1.09.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=None,
    sep="\t",
)
hindi_PQ64.columns = ["margin score", "lang1", "lang2"]
print(hindi_PQ64["margin score"].describe())

print("hindi_PQ512:")
hindi_PQ512 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/faiss/hin/eng0-hin.TH-1.TH-1.13.bitext.TH-1.TH-1.13.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=None,
    sep="\t",
)
hindi_PQ512.columns = ["margin score", "lang1", "lang2"]
print(hindi_PQ512["margin score"].describe())

print("slovak_PQ64:")
slk_PQ64 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/baseline/slk/eng0-slk.TH-1.TH-1.09.bitext.TH-1.TH-1.09.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=None,
    sep="\t",
)
slk_PQ64.columns = ["margin score", "lang1", "lang2"]
print(slk_PQ64["margin score"].describe())

print("slovak_PQ512:")
slk_PQ512 = pd.read_csv(
    "/checkpoints/kevinheffernan/seamless/nmt.23h1.seamless-faiss/faiss/slk/eng0-slk.TH-1.TH-1.13.bitext.TH-1.TH-1.13.bitext.tsv.gz",
    compression="gzip",
    on_bad_lines="skip",
    header=0,
    sep="\t",
)
slk_PQ512.columns = ["margin score", "lang1", "lang2"]
print(slk_PQ512["margin score"].describe())

# for a given threshold, check the overlap of pairs between the baseline and PQ512
