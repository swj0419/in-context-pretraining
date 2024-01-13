from jsonlines import Writer
from typing import Optional
from pathlib import Path
import numpy as np
import torch

PAD_TOKEN = 0
CLS_TOKEN = 101
SEP_TOKEN = 102
ROLL_EVERY = 4096
TOKENIZER = []
MODEL = None
CASED = True


def exists(x):
    return x is not None


def get_tokenizer(
    name: Optional[str] = None,
    # TODO: apply this to retro_z_data
    # copied from ~/.cache/toch/hub
    repo_or_dir: str = "/checkpoint/hcir/torch/hub/huggingface_pytorch-transformers_main",
    source: str = "local",
    skip_validation: bool = False,
):
    if name is None:
        name = f'bert-base-{"" if CASED else "un"}cased'
    if source == "local":
        repo_or_dir = str(Path(repo_or_dir).expanduser())

    if len(TOKENIZER) == 0:
        TOKENIZER.append(torch.hub.load(repo_or_dir, "tokenizer", name, skip_validation=skip_validation, source=source))
    return TOKENIZER[0]


def tokens(x):
    return get_tokenizer().convert_ids_to_tokens(x)


def ids_to_string(x):
    return get_tokenizer().convert_tokens_to_string(tokens(x))  # noqa: E731


@torch.no_grad()
def bert_embed(token_ids, return_cls_repr=False, eps=1e-8, pad_id=0.0, model_config=None):
    if model_config is None:
        model = get_bert()
    else:
        model = get_bert(**model_config)
    mask = token_ids != pad_id

    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    outputs = model(input_ids=token_ids, attention_mask=mask, output_hidden_states=True)  # type: ignore

    hidden_state = outputs.hidden_states[-1]

    if return_cls_repr:
        return hidden_state[:, 0]  # return [cls] as representation

    if not exists(mask):
        return hidden_state.mean(dim=1)

    mask = mask[:, 1:]  # mean all tokens excluding [cls], accounting for length
    mask = mask.unsqueeze(dim=2)  # rearrange(mask, "b n -> b n 1")

    numer = (hidden_state[:, 1:] * mask).sum(dim=1)
    denom = mask.sum(dim=1)
    masked_mean = numer / (denom + eps)
    return masked_mean


def _embed_chunk_batch(chunk_batch, model):
    padded_batch = np.concatenate((chunk_batch, np.full((chunk_batch.shape[0], 2), PAD_TOKEN)), axis=1)
    for index, _ in enumerate(padded_batch):
        if padded_batch[index, 0] != CLS_TOKEN:
            padded_batch[index] = np.roll(padded_batch[index], 1)
            padded_batch[index, 0] = CLS_TOKEN

        if SEP_TOKEN not in padded_batch[index]:
            pad_indices = np.where(padded_batch[index] == PAD_TOKEN)
            assert len(pad_indices) == 1
            pad_indices = pad_indices[0]
            padded_batch[index, pad_indices[0]] = SEP_TOKEN

    batch_embed = bert_embed(torch.from_numpy(padded_batch), model_config=model)
    return batch_embed


def get_bert(
    name: Optional[str] = None,
    repo_or_dir: str = "/checkpoint/hcir/torch/hub/huggingface_pytorch-transformers_main",
    source: str = "local",
    skip_validation: bool = False,
):
    if name is None:
        name = f'bert-base-{"" if CASED else "un"}cased'
    if source == 'local':
        repo_or_dir = str(Path(repo_or_dir).expanduser())
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load(
            repo_or_dir,
            "model",
            name,
            skip_validation=skip_validation,
            source=source,
        )
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL


class ChunkLoggerDummy(object):
    def __init__(self, chunk_len: int, seq_len: int, k_: int, log_dir: Path, filename_root):
        pass

    def log_example(self, example_dict, flush: bool = False):
        pass


def l2_dist(a0, a1):
    return torch.cdist(a0.unsqueeze(dim=0), a1.unsqueeze(dim=0))


# to format jsonl files:
# cat test.jsonl | sed -e 's/\"neighbor_00\(.\)\"/\n\t\"neighbor_00\1\"/g' -e  's/\"neighbor_00\(.\)_cos\"/\n\t\"neighbor_00\1_cos\"/g' -e  's/\"continuation_00\(.\)\"/\n\t\"continuation_00\1\"/g' -e  's/\"continuation_00\(.\)_cos\"/\n\t\"continuation_00\1\_cos"/g'
class ChunkLogger(object):
    def __init__(self, chunk_len: int, seq_len: int, k_: int, log_dir: Path, shard_index: int):
        self.chunk_len = chunk_len
        self.seq_len = seq_len
        self.k = k_
        self.filename_root = f'{shard_index:>05d}'
        self.log_dir = log_dir

        self.chunks_per_seq, mod = divmod(seq_len, chunk_len)
        assert mod == 0, f'Invalid mod: {mod}'

        self.num_chunks_processed = 0

        self.next_file_number = 0
        self.current_file = None
        self.writer = None
        self._roll_file()

        self.cos = torch.nn.CosineSimilarity(dim=0)

    def _roll_file(self):
        if self.current_file is not None:
            self.current_file.close()
        file_path = self.log_dir / f'{self.filename_root}_{self.next_file_number:>04d}.jsonl'
        self.current_file = open(file_path, 'w')
        self.writer = Writer(self.current_file)
        self.next_file_number += 1

    def _get_cosines_and_l2s(self, chunk_ids, neighbor_ids, continuation_ids):
        stack = np.stack((chunk_ids, neighbor_ids, continuation_ids))
        embeddings = _embed_chunk_batch(stack, None)
        chunk_emb, neighbor_emb, continuation_emb = embeddings

        cosines = self.cos(chunk_emb, neighbor_emb), self.cos(chunk_emb, continuation_emb)
        l2s = l2_dist(chunk_emb, neighbor_emb), l2_dist(chunk_emb, continuation_emb)
        return cosines, l2s

    def _create_obj(self, chunk_ids: np.ndarray, chunk_neighbor_tokens: np.ndarray):
        chunk_str = ids_to_string(chunk_ids)
        chunk_obj = {'id': self.num_chunks_processed, 'text': chunk_str}
        for neighbor_index, neighbor in enumerate(chunk_neighbor_tokens):
            neighbor_ids = neighbor[:self.chunk_len]
            neighbor_str = ids_to_string(neighbor_ids)
            continuation_ids = neighbor[self.chunk_len:]
            continuation_str = ids_to_string(continuation_ids)
            cosines, l2s = self._get_cosines_and_l2s(chunk_ids, neighbor_ids, continuation_ids)
            neighbor_cos, continuation_cos = cosines
            neighbor_l2, continuation_l2 = l2s
            chunk_obj[f'neighbor_{neighbor_index:0>3}'] = neighbor_str
            chunk_obj[f'neighbor_{neighbor_index:0>3}_cos'] = neighbor_cos.item()
            chunk_obj[f'neighbor_{neighbor_index:0>3}_l2'] = neighbor_l2.item()
            chunk_obj[f'continuation_{neighbor_index:0>3}'] = continuation_str
            chunk_obj[f'continuation_{neighbor_index:0>3}_cos'] = continuation_cos.item()
            chunk_obj[f'continuation_{neighbor_index:0>3}_l2'] = continuation_l2.item()

        return chunk_obj

    def _log_chunk(self, chunk_tokens: np.ndarray, chunk_neighbor_tokens: np.ndarray, flush: bool):
        chunk_obj = self._create_obj(chunk_tokens, chunk_neighbor_tokens)
        self.writer.write(chunk_obj)  # type: ignore
        if flush:
            self.current_file.flush()  # type: ignore
        self.num_chunks_processed += 1
        if ((self.num_chunks_processed + 1) % ROLL_EVERY) == 0:
            self._roll_file()

    def log_example(self, example_dict, flush: bool = False):
        example_tokens = example_dict["example_tokens"]
        neighbor_tokens = example_dict["neighbor_tokens"]
        assert example_tokens.shape == (self.seq_len,)
        assert neighbor_tokens.shape == (self.k, self.chunks_per_seq, self.chunk_len * 2)
        for chunk_index in range(self.chunks_per_seq):
            tokens_slice = slice(chunk_index * self.chunk_len, (chunk_index + 1) * self.chunk_len)
            chunk_tokens = example_tokens[tokens_slice]
            chunk_neighbor_tokens = neighbor_tokens[:, chunk_index, :]
            self._log_chunk(chunk_tokens, chunk_neighbor_tokens, flush)
