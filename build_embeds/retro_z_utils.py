from contextlib import contextmanager
from rich.logging import RichHandler
from pathlib import Path
import numpy as np
import lz4.frame
import jsonlines
import logging
# from ipdb import set_trace as bp


def exists_and_is_dir(file_path):
    exists = file_path.exists()
    if not exists:
        logging.critical(f'File {file_path} does not exist')
    is_dir = file_path.is_dir()
    if not is_dir:
        logging.critical(f'File {file_path} is not a directory')

    return exists and is_dir


def exists_and_is_file(file_path):
    exists = file_path.exists()
    if not exists:
        logging.critical(f'File {file_path} does not exist')
    is_file = file_path.is_file()
    if not is_file:
        logging.critical(f'{file_path} is not a file')

    return exists and is_file


# TODO: make file handler log timestamp (sigh) https://stackoverflow.com/questions/33567386/python-logging-not-writing-timestamp-to-output-file
def init_logging() -> None:
    # bp()
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[RichHandler(rich_tracebacks=True)],
                        # TODO: this would get completely jacked by distribution
                        # logging.FileHandler(f'/checkpoint/{os.getlogin()}/logs/retro-z/retro_z_data.log')
                        force=True)
    

@contextmanager
def log(msg: str, level: int = logging.INFO):
    logging.log(level, msg)
    yield None
    logging.log(level, f'{msg} completed')


@contextmanager
def read_jsonl_file(file_path: Path):
    with lz4.frame.open(file_path, 'rb') as file_:
        with jsonlines.Reader(file_) as reader:  # type: ignore
            yield reader


@contextmanager
def read_jsonl_file_no_compress(file_path: Path):
    with open(file_path, 'r') as file:
        with jsonlines.Reader(file) as reader:  # type: ignore
            yield reader


@contextmanager
def write_jsonl_file(file_path: Path):
    with lz4.frame.open(file_path, 'wb') as file_:
        with jsonlines.Writer(file_) as writer:  # type: ignore
            yield writer


@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference


# TODO: rename
# TODO: consider using more than in just knns
# don't go over the end in the (probable) case that length isn't batch_size-aligned
def range_chunked(max_value: int, batch_size: int, min_value: int = 0, exact: bool = False):
    assert not exact or ((max_value - min_value) % batch_size == 0), f'Requested exact chunked range but parameters invalid: max_value {max_value}; min_value {min_value}; batch_size {batch_size}'

    start = min_value
    while start < max_value:
        end = min(start + batch_size, max_value)
        yield slice(start, end)
        start = end


def reshape_memmap_given_width(flat, second_dim):
    first_dim, mod = divmod(len(flat), second_dim)
    assert first_dim > 0, f'first_dim is 0 in reshape_memmap_given_width with array shape {flat.shape} and second_dim {second_dim}'
    assert mod == 0, f'mod is {mod} in reshape_memmap_given_width with array shape {flat.shape} and second_dim {second_dim}'
    return flat.reshape(first_dim, second_dim), first_dim
