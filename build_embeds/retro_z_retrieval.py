from typing import List, Tuple
import numpy as np
import logging

INVALID_CONTINUATION_TOKEN = 0


class TetrisArray(object):
    '''allow the creation of a 2D array where coumns are added to piecemeal, in a ragged manner'''
    def __init__(self, k_: int, num_chunks: int, chunk_len: int):
        assert num_chunks > 0, f'Invalid num_chunks: {num_chunks}'
        assert chunk_len > 0, f'Invalid chunk_len: {chunk_len}'
        assert k_ > 0, f'Invalid k_: {k_}'

        self._chunk_len = chunk_len
        # allocate entire result set. chunk_len * 2 to account for continuations, as well as neighbors
        self.tokens = np.zeros((k_, num_chunks, self._chunk_len * 2), dtype=np.int32)
        # we're going to gradually populate the result by iterating over the chunks/tokens files, grabbing chunks that
        # correspond to neighbors. we'll be filling the results array in a tetris-like manner and need to keep note of
        # the index at which to insert the next "batch" of neighbors (from the current chunks file)
        self._next_k_index_for_chunk = np.zeros((num_chunks,), dtype=np.int32)

    def store(self, chunk_index: int, neighbors_tokens: np.ndarray, continuation_tokens: np.ndarray) -> None:
        assert chunk_index >= 0, f'Invalid chunk_index: {chunk_index}'

        start_k_index = self._next_k_index_for_chunk[chunk_index]
        end_k_index = start_k_index + len(neighbors_tokens)
        assert end_k_index <= self.tokens.shape[0], f'Invalid end_k_index ({end_k_index} or self.tokens.shape[0] ({self.tokens.shape[0]})'
        self.tokens[start_k_index:end_k_index, chunk_index, :] = np.concatenate((neighbors_tokens, continuation_tokens),
                                                                                axis=1)
        self._next_k_index_for_chunk[chunk_index] = end_k_index


def _get_neighbors_and_tokens(chunks_tokens, chunk_min_index, chunk_max_index, chunk_neighbors, k_chunk_neighbors_mask):
    in_memmap = (chunk_neighbors >= chunk_min_index) & (chunk_neighbors < chunk_max_index)
    neighbors_in_memmap_absolute = chunk_neighbors[in_memmap]
    k_chunk_neighbors_mask_in_memmap = k_chunk_neighbors_mask[in_memmap]
    # join on chunks to get neighbor tokens
    neighbors_relative = neighbors_in_memmap_absolute - chunk_min_index
    neighbors_tokens_unfiltered = chunks_tokens[neighbors_relative]
    # TODO: .T is to make broadcasting work - make this more elegant
    neighbors_tokens = np.where(~k_chunk_neighbors_mask_in_memmap.T, 0, neighbors_tokens_unfiltered.T)

    return neighbors_in_memmap_absolute, neighbors_tokens.T


def _get_continuations_tokens(chunks_tokens, neighbors_absolute, chunk_min_index, chunks_to_documents, chunk_max_index):
    # we need to add continuations only if they're in the same doc, so we'll capture that condition for later
    # we also need to be careful not to index out of bounds of current chunk, so we will clip the continuation index
    # we'll address both of these cases later and zero out the corresponding (invalid) tokens
    continuations_absolute = neighbors_absolute + 1
    continuations_absolute_clipped = continuations_absolute.clip(max=chunk_max_index - 1)
    docs_of_neighbors = chunks_to_documents[neighbors_absolute]
    docs_of_continuations = chunks_to_documents[continuations_absolute_clipped]

    # cleaner to zero invalid continuations after populating them so we'll proceed here as if continuations are all valid
    continuations_relative = continuations_absolute_clipped - chunk_min_index
    continuations_tokens = chunks_tokens[continuations_relative]

    # now filter out continuations that were subject to clipping in either of the two cases above
    invalid_continuation = np.expand_dims((docs_of_continuations != docs_of_neighbors) |  # noqa: W504
                                          (neighbors_absolute == continuations_absolute_clipped), axis=1)
    continuations_filtered = np.where(invalid_continuation, INVALID_CONTINUATION_TOKEN, continuations_tokens)

    return continuations_filtered


def _subset_neighbors(num_chunks: int, k_: int, neighbors_of_chunks: np.memmap, neighbors_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert num_chunks > 0, f'Invalid num_chunks: {num_chunks}'
    assert k_ > 0, f'Invalid k_: {k_}'
    # for each chunk, subset neighbors to just the k we need, ignoring those that share a doc with
    # the chunk in question. an intermediate result is a ragged array so we need the loop
    k_neighbors_of_chunks = np.zeros((num_chunks, k_), dtype=np.uint32)
    k_neighbors_mask = np.zeros((num_chunks, k_), dtype=np.bool)  # type: ignore
    for chunk_index, neighbors_of_chunk in enumerate(neighbors_of_chunks):
        chunk_neighbors_mask = neighbors_mask[chunk_index]
        chunk_neighbors_filtered = neighbors_of_chunk[chunk_neighbors_mask]
        k_chunk_neighbors = chunk_neighbors_filtered[:k_]
        # :len ensures we correctly handle the case that, after masking, we have fewer than k_ neighbors
        k_neighbors_of_chunks[chunk_index, :len(k_chunk_neighbors)] = k_chunk_neighbors
        k_neighbors_mask[chunk_index, :len(k_chunk_neighbors)] = True
        if(len(k_chunk_neighbors) < k_):
            logging.warning(f'Less than k neighbors remain after discarding neighbors from the same doc: {len(k_chunk_neighbors)}')

    logging.debug(f'neighbors_of_chunks[:4] vs. k_neighbors_of_chunks[:4], for each chunk:\n'
                  f'{[str(neighbors_of_chunks[i,: 4]) + " vs. " + str(k_neighbors_of_chunks[i,: 4]) for i in range(num_chunks)]}')

    return k_neighbors_of_chunks, k_neighbors_mask


def _retrieve_tokens(k_: int, num_chunks: int, chunk_len: int, chunks_memmaps: List[np.memmap],
                     indices_of_memmaps: List[Tuple[int, int, int]], k_neighbors_of_chunks: np.ndarray,
                     chunks_to_docs: np.memmap, k_neighbors_mask: np.ndarray) -> np.ndarray:
    assert k_neighbors_of_chunks.shape == (num_chunks, k_), f'Invalid shape of k_neighbors_of_chunks: {k_neighbors_of_chunks.shape}'
    assert len(chunks_memmaps) == len(indices_of_memmaps), 'Mismatchings lengths of chunks_memmaps and indices_of_memmaps'
    # assert num_chunks > 0 and num_chunks == 32, f'Invalid num_chunks: {num_chunks}'
    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'
    assert k_ > 0, f'Invalid k_: {k_}'

    retrieved_tokens = TetrisArray(k_, num_chunks, chunk_len)
    # loop over the chunks arrays and then, for each chunks array index over the columns representing queries and neighbors
    # chunks_tokens to be read as chunks' tokens, i.e. tokens of chunks [in an array]
    for chunks_memmap_index, chunks_memmap in enumerate(chunks_memmaps):
        chunk_min_index, chunk_max_index, memmap_order = indices_of_memmaps[chunks_memmap_index]
        assert chunk_min_index >= 0 and chunk_max_index >= 0, f'Invalid chunk_min_index ({chunk_min_index}) or chunk_max_index ({chunk_max_index})'
        assert memmap_order == chunks_memmap_index, f'Invalid memmap_order ({memmap_order}) or chunks_memmap_index ({chunks_memmap_index})'

        for chunk_index, neighbors_of_chunk in enumerate(k_neighbors_of_chunks):
            k_chunk_neighbors_mask = k_neighbors_mask[chunk_index]

            neighbors_absolute, neighbors_tokens = \
                _get_neighbors_and_tokens(chunks_memmap, chunk_min_index, chunk_max_index, neighbors_of_chunk,
                                          k_chunk_neighbors_mask)

            # don't waste time looking for continuations or trying to store tokens if we don't have any neighbors in this memmap
            if len(neighbors_absolute) > 0:
                continuation_tokens = _get_continuations_tokens(chunks_memmap, neighbors_absolute,
                                                                chunk_min_index, chunks_to_docs, chunk_max_index)
                retrieved_tokens.store(chunk_index, neighbors_tokens, continuation_tokens)

    return retrieved_tokens.tokens


def get_neighbors_and_continuations(chunks_ids_absolute: np.ndarray, neighbors_of_chunks: np.memmap,
                                    chunks_memmaps: List[np.memmap], chunk_ids_to_doc_ids: np.memmap,
                                    indices_of_memmaps: List[Tuple[int, int, int]], chunk_len: int, k_: int, is_eval_dataset: bool) -> np.ndarray:
    '''chunks_ids_absolute:     1D array of ids of chunks for which we're retrieving neighbors and continuations
       neighbors_of_chunks:     2D array of K neighbor ids (rows) for a given chunk (columns)
       chunks_memmaps:          list of memory mapped 2D arrays of chunk ids (rows) to tokens (columns)
       chunk_ids_to_doc_ids:    map from a given chunk id to the id of the document from which it originates
       indices_of_memmaps:      list of the relevant indices for a given memmap, viz: first chunk id, last chunk id, index in list
       chunk_len:               the length of a chunk (e.g. 64 tokens)
       k_:                      number of neighbors to return tokens for (as distinct from K, the number under consideration'''
    assert chunk_len > 0, f'Invalid chunk_len: {chunk_len}'
    assert k_ > 0, f'Invalid k_: {k_}'
    # we don't get k from here since the neighbor dimension here is typically larger than we need (K > k_)
    num_chunks, K = neighbors_of_chunks.shape  # noqa: N806

    # we may not care about chunk->doc mapping if we're doing eval, since a neighbor can't be defn. be from the same doc
    if not is_eval_dataset:
        # determine which neighbors share a document with their corresponding chunks so we can mask them out later
        docs_of_chunks = chunk_ids_to_doc_ids[chunks_ids_absolute]
        docs_of_neighbors = chunk_ids_to_doc_ids[neighbors_of_chunks]
        neighbors_mask = docs_of_neighbors != docs_of_chunks[:, np.newaxis].repeat(K, axis=1)
    else:
        neighbors_mask = np.full((len(chunks_ids_absolute), K), True, dtype=np.bool)

    logging.debug(f"Neighbors masks' sums for chunks: {[neighbors_mask[i].sum() for i in range(num_chunks)]}")

    k_neighbors_of_chunks, k_neighbors_mask = _subset_neighbors(num_chunks, k_, neighbors_of_chunks, neighbors_mask)
    # now we have valid neighbors of the correct number, we can get tokens for neighbors and continuations
    retrieved_tokens = _retrieve_tokens(k_, num_chunks, chunk_len, chunks_memmaps, indices_of_memmaps,
                                        k_neighbors_of_chunks, chunk_ids_to_doc_ids, k_neighbors_mask)

    return retrieved_tokens


# TODO: this does not work at present in the eval case where chunks->docs map is None
def get_last_chunk_of_doc_flags(chunks_ids_absolute: np.ndarray, chunk_ids_to_doc_ids: np.memmap):
    '''chunks_ids_absolute:     ids of chunks
       chunk_ids_to_doc_ids:    map from a given chunk id to the id of the document from which it originates'''

    # determine whether a given chunk is the last in its doc
    # (which which case the trainer will likely not want to pass its neighbor or continuation tokens to the model)
    first_chunk_index, last_chunk_index = chunks_ids_absolute[0], chunks_ids_absolute[-1]
    chunks_to_docs_subset = np.append(chunk_ids_to_doc_ids[first_chunk_index: (last_chunk_index + 1) + 1], -1)
    chunks_relative = chunks_ids_absolute - first_chunk_index
    last_chunk_of_doc_flags = chunks_to_docs_subset[chunks_relative] != chunks_to_docs_subset[chunks_relative + 1]
    logging.debug(f'Last chunk of doc flags:\n{last_chunk_of_doc_flags}')

    return last_chunk_of_doc_flags
