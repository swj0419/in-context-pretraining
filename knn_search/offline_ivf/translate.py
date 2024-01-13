import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from multiprocessing.pool import Pool
import torch
import submitit

def translate_on_gpu(param):
    gpu_id, sents = param
    print("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-3.3B", src_lang="fin_Latn"
    )
    print("loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    print("copy model to gpu")
    model_gpu = model.to(f'cuda:{gpu_id}')
    with torch.no_grad():
        trs = []
        for sent in sents:
            #print(f"{sent=}")
            inputs = tokenizer(sent, return_tensors="pt")
            inputs_gpu = inputs.to(f'cuda:{gpu_id}')
            translated_tokens = model_gpu.generate(
                **inputs_gpu, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=300
            )
            tr = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            #print(f"{tr=}")
            trs.append(tr)
        assert len(sents) == len(trs)
        return trs

def translate_on_node(sents):
    nt = torch.cuda.device_count()
    if nt == 1:
        return translate_on_gpu((0, sents))
    params = []
    for i in range(nt):
        i0, i1 = i * len(sents) // nt, (i + 1) * len(sents) // nt
        params.append((i, sents[i0:i1]))
    pool = Pool(nt)
    trs = [''] * len(sents)
    for i, tr in enumerate(list(pool.map(translate_on_gpu, params))):
        i0, i1 = i * len(sents) // nt, (i + 1) * len(sents) // nt
        trs[i0:i1] = tr
    return trs

def launch_jobs(params, local=True):
    if local:
        results = [translate_on_node(p) for p in params]
        return results
    print(f'launching {len(params)} jobs')
    executor = submitit.AutoExecutor(folder='/checkpoint/gsz/jobs')
    executor.update_parameters(
        nodes=1,
        gpus_per_node=1,
        cpus_per_task=10,
        mem_gb=64,
        tasks_per_node=1,
        name="translate",
        slurm_array_parallelism=512,
        slurm_partition="scavenge",
        slurm_time=1 * 60,
        slurm_constraint="volta32gb",
    )
    jobs = executor.map_array(translate_on_node, params)
    print(f'launched {len(jobs)} jobs')
    results = [job.result() for job in jobs]
    print(f'received {len(results)} results')
    return results

if __name__ == "__main__":
    for nrows in range(723866, 723867, 80000):
        print("loading sents")
        df_st = pd.read_csv("/checkpoint/gsz/seamless/fin_sents.tsv", header=None, sep="\t", nrows=nrows)
        try:
            df_tr = pd.read_csv("/checkpoint/gsz/seamless/fin_eng_sents.tsv", header=None, sep="\t")
        except pd.errors.EmptyDataError:
            df_tr = pd.DataFrame([['[DUMMY]']], columns=[0])

        print(df_st)

        print("filtering sents")
        df = df_st.merge(df_tr, how='left', on=[0])

        print(df)

        df = df[df[1].isnull()].reset_index()[[0]]

        print(df)

        if len(df) > 0:
            bs = 500
            trans_input = []
            for i in range(0, len(df), bs):
                j = min(i + bs, len(df))
                trans_input.append(df.loc[i:j, 0].tolist())
            
            trans_output = launch_jobs(trans_input, False)

            k = 0
            for i in range(0, len(df), bs):
                j = min(i + bs, len(df))
                df.loc[i:j, 1] = trans_output[k]
                k += 1

            df[[0,1]].to_csv("/checkpoint/gsz/seamless/fin_eng_sents.tsv", mode='a', sep="\t", header=None, index=False)
        else:
            print('nothing to do')

        print(df)
