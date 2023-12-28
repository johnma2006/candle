"""Generation algorithms."""

import numpy as np
from typing import List, Callable

from ..tensor import Tensor
import candle
import candle.functions as F


def batch_generation(model, 
                     indices: Tensor,
                     n_tokens_to_gen: int,
                     top_k: int = None,
                     top_p: float = None,
                     temperature: float = 1.0,
                     sample: bool = True,
                     use_kv_cache: bool = True):
    """Generates batch of responses with KV caching.
    
    NOTE: If KV caching, we will guarantee that:
        (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
    This will be true even if the generator terminates early, as long as generator.close() is called

    Parameters
    ----------
    model
        Model instance. model should take in tensors of shape (batch, seqlen) and output logits of
        shape (batch, seqlen, vocab_size).
    indices
        Conditioning sequence. Tensor of shape (batch, seqlen).
    n_tokens_to_gen
        Number of tokens to generate.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities
        False to take argmax.
    use_kv_cache
        If True, uses kv_cache to speed up inference. If generator terminates early, make sure to call
        generator.close() to properly maintain the KV cache.
        
        After generation, we will guarantee that:
            (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices) + len(response)
        
    Returns
    -------
    generator
        The generator will yield tokens as soon as they are sampled.
        
    Examples
    --------
    (Pseudocode) In the following, response2 == kv_response2 while being faster to generate.

        prompt1 = 'Hi, I am John.'
        prompt2 = 'That is great.'

        response1 = batch_generation(prompt1, use_kv_cache=False)
        response2 = batch_generation(prompt1 + response1 + prompt2, use_kv_cache=False)

        response1 = batch_generation(prompt1, use_kv_cache=True)
        kv_response2 = batch_generation(prompt2, use_kv_cache=True)

    """
    model.eval()

    for token_n in range(n_tokens_to_gen):
        if indices.shape[1] >= model.block_size:
            raise RuntimeError(f'Conversation has reached the limit of {model.block_size} tokens.')

        with candle.no_grad():
            indices_to_input = indices
            if use_kv_cache:
                # After the first step, feed in one token at a time
                if token_n > 0:
                    indices_to_input = indices_to_input[:, -1:]

            next_token_logits = model(indices_to_input, use_kv_cache)[:, -1]

        probs = F.softmax(next_token_logits / (temperature + 1e-6)).data.T  # shape (vocab_size, batch)
        (vocab_size, batch) = probs.shape

        if top_k is not None:
            probs = top_k_sample(probs, top_k)

        if top_p is not None:
            probs = nucleus_sample(probs, top_p)

        if sample:
            next_indices = [np.random.choice(range(vocab_size), p=probs[:, i])
                            for i in range(batch)]
        else:
            next_indices = np.argmax(probs, axis=0).tolist()

        next_indices_tensor = Tensor(np.array(next_indices)[:, None])

        try:
            yield next_indices
        except GeneratorExit:
            # This means that the generator exited early. We have to feed the last
            # generated indices back in to maintain the KV cache
            if use_kv_cache:
                _ = model(next_indices_tensor, use_kv_cache=True)
            return

        indices = F.concat([indices, next_indices_tensor], axis=1)

    # We have to feed the last generated indices back in to maintain the KV cache
    if use_kv_cache:
        _ = model(next_indices_tensor, use_kv_cache=True)
        
        
def top_k_sample(probs: np.array,
                 top_k: int):
    """Top-k sampling.
    
    Parameters
    ----------
    probs
        Numpy array of probabilities with shape (vocab_size, batch).
        Modifies probs in place.
    top_k
        Top K words to filter to.
    
    """
    # For each row, zero out everything except for top_k probs per row
    top_k_prob = np.sort(probs, axis=0)[-top_k, :]
    probs[probs < top_k_prob] = 0
    
    probs /= probs.sum(axis=0)
    
    return probs


def nucleus_sample(probs: np.array,
                   top_p: int):
    """Nucleus sampling. Filter to top probs such that the sum prob is just less than top_p.
    
    References:
    [1] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.
        The Curious Case of Neural Text Degeneration. arXiv:1904.09751, 2019
    
    Parameters
    ----------
    probs
        Numpy array of probabilities with shape (vocab_size, batch).
        Modifies probs in place.
    top_p
        Filter to the top `k` probs such that the sum probs is <= top_p and k is largest.
    
    """
    sorted_probs = np.sort(probs, axis=0)[::-1]
    cum_probs = sorted_probs.cumsum(axis=0)
    top_k = (cum_probs <= top_p).sum(axis=0)

    ranking = probs.shape[0] - np.argsort(np.argsort(probs, axis=0), axis=0)
    mask = (ranking <= top_k) | (ranking == 1)  # | (ranking == 1) accounts for when the edge case if highest prob > top_p

    probs[~mask] = 0
    probs /= probs.sum(axis=0)

    return probs


def beam_search_generation(model, 
                           indices: Tensor,
                           n_tokens_to_gen: int,
                           beam_size: int,
                           top_k: int = None,
                           top_p: float = None,
                           temperature: float = 1.0,
                           sample: bool = True,
                           use_kv_cache: bool = True,
                           modify_kv_cache_func: Callable = None):
    """Generates tokens using beam search with KV caching.
    
    Tokens will be yielded as soon as all `beam_size` beams agree on that token.
    
    NOTE: If KV caching, we will guarantee that:
        (kv_cache_seqlen after) == (kv_cache_seqlen before) + len(indices)
    Note this does NOT include the generated sequence.
    This will be true even if the generator terminates early, as long as generator.close() is called

    Parameters
    ----------
    model
        Model instance. model should take in tensors of shape (batch, seqlen) and output logits of
        shape (batch, seqlen, vocab_size).
    indices
        Conditioning sequence. Tensor of shape (seqlen,).
    n_tokens_to_gen
        Number of tokens to generate.
    beam_size
        Beam size to use in beam search. e.g., beam_size = 1 for greedy search.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities
        False to take argmax.
    use_kv_cache
        If True, uses kv_cache to speed up inference. If generator terminates early, make sure to call
        generator.close() to properly maintain the KV cache.
        
        After generation, we will guarantee that:
            (kv_cache_seqlen after decoding) == (kv_cache_seqlen before decoding) + len(indices)
        
        If using KV caching, then must pass in `modify_kv_cache_func`.
    modify_kv_cache_func
        Function with signature `modify_kv_cache_func(model, trim_seqlen, reindex_batch_indices),
        that modifies the model's kv_cache by trimming or reindexing.

        See `default_modify_kv_cache` for example implementation.
        
    Returns
    -------
    generator
        The generator will yield List[int] of tokens as soon as all `beam_size` beams
        agree on the tokens.
        
    Examples
    --------
    (Pseudocode) In the following, response2 == kv_response2 while being faster to generate.

        prompt1 = 'Hi, I am John.'
        prompt2 = 'That is great.'

        response1 = beam_search_generation(prompt1, use_kv_cache=False)
        response2 = beam_search_generation(prompt1 + response1 + prompt2, use_kv_cache=False)

        response1 = beam_search_generation(prompt1, use_kv_cache=True)
        kv_response2 = beam_search_generation(response1 + prompt2, use_kv_cache=True)

    """
    model.eval()

    if modify_kv_cache_func is None: 
        modify_kv_cache_func = default_modify_kv_cache
    if beam_size is None:
        beam_size = 1
    if use_kv_cache:
        final_kv_cache_len = model.get_kv_cache_seqlen() + len(indices)

    # We maintain `cumulative_log_prob_per_beam` to have the same length as `indices`
    # On the 0th iter, `indices` has shape (1, initial_seqlen).
    # Afterwards, `indices` has shape (beam_size, initial_seqlen + iter).
    cumulative_log_prob_per_beam = np.array([0.0])
    indices = Tensor(indices.data[None, :])

    # indices[:, head_index] is the first token that hasn't been yielded yet
    head_index = indices.shape[1]

    for token_n in range(n_tokens_to_gen):
        if indices.shape[1] >= model.block_size:
            raise RuntimeError(f'Conversation has reached the limit of {model.block_size} tokens.')
            
        with candle.no_grad():
            indices_to_input = indices
            if use_kv_cache:
                # After the first step, feed in one token at a time
                if token_n > 0:
                    indices_to_input = indices_to_input[:, -1:]

            next_token_logits = model(indices_to_input, use_kv_cache)[:, -1]

        probs = F.softmax(next_token_logits / temperature).data.T  # shape (vocab_size, beam_size)

        if top_k is not None:
            probs = top_k_sample(probs, top_k)

        if top_p is not None:
            probs = nucleus_sample(probs, top_p)

        # Accumulate prob by `cumulative_log_prob_per_beam`
        probs += 1.0e-3 / len(probs)  # Add small eps prob to guarantee num_non_zero(probs) >= beam_size
        log_probs = np.log(probs) + cumulative_log_prob_per_beam

        # `next_beam_indices` are the best `beam_size` candidates
        # Each `beam_index` in `next_beam_indices` is defined as:
        #    beam_index // indices.shape[0] is the token index of the next token
        #    indices[beam_index % indices.shape[0]] is the beam that token will be added to
        if sample:
            normalized_probs = candle.utils.softmax(log_probs.flatten())
            next_beam_indices = np.random.choice(np.arange(log_probs.size),
                                                 size=beam_size,
                                                 p=normalized_probs,
                                                 replace=False)
        else:
            next_beam_indices = log_probs.flatten().argsort()[-beam_size:]

        cumulative_log_prob_per_beam = log_probs.flatten()[next_beam_indices]

        # Update indices by appending the tokens from next_beam_indices
        new_indices = np.zeros((beam_size, indices.shape[1] + 1)).astype(int)
        reindex_kv_indices = []
        for (i, beam_index) in enumerate(next_beam_indices):
            next_index = beam_index // indices.shape[0]
            indices_i = beam_index % indices.shape[0]

            print(indices)
            print(type(indices_i))
            print(repr(indices_i))
            print(indices_i)
            print(indices_i.shape)
            new_indices[i] = np.append(indices[indices_i].data, next_index)
            reindex_kv_indices.append(indices_i)
        indices = Tensor(new_indices)

        # Realign kv_cache with new beam search indices
        if use_kv_cache:
            modify_kv_cache_func(model, reindex_batch_indices=reindex_kv_indices)

        # If all `beam_index` beams have the same token at head_index, then it's
        # safe to yield that token
        indices_at_head = indices.data[:, head_index]
        while np.unique(indices_at_head).size == 1:
            try:
                head_index += 1
                yield [int(indices_at_head[0])]

            except GeneratorExit:
                # This means that the generator exited early. We have to do some cleanup to
                # maintain the KV cache.
                if use_kv_cache:
                    # Trim the KV cache to guarantee that
                    # (kv_cache_seqlen after decoding) == final_kv_cache_len
                    modify_kv_cache_func(model,
                                         trim_seqlen=final_kv_cache_len,
                                         reindex_batch_indices=[0])
                return

            if head_index == indices.shape[1]:
                break

            indices_at_head = indices.data[:, head_index]

    # Yield rest of the generated indices
    if sample:
        best_index = np.random.choice(np.arange(indices.shape[0]),
                                      p=candle.utils.softmax(cumulative_log_prob_per_beam))
    else:
        best_index = cumulative_log_prob_per_beam.argmax()

    # Trim the KV cache to guarantee that (kv_cache_seqlen after decoding) == final_kv_cache_len
    if use_kv_cache:
        modify_kv_cache_func(model,
                             trim_seqlen=final_kv_cache_len,
                             reindex_batch_indices=[best_index])

    yield indices.data[best_index, head_index:].astype(int).tolist()


def default_modify_kv_cache(model,
                            trim_seqlen: int = None,
                            reindex_batch_indices: List[int] = None):
    """Modifies the kv_cache by trimming or reindexing.

    Parameters
    ----------
    trim_seqlen
        Trims kv_cache along the seqlen dimension to length `new_seqlen`, keeping earlier tokens.
    batch_indices
        Reindexes the kv_cache along the batch dimension. Necessary during beam search.

    """
    for decoder_block in model.decoder_blocks:
        if decoder_block.attn.kv_cache is not None:
            (key_cache, value_cache) = decoder_block.attn.kv_cache

            if trim_seqlen is not None:
                if trim_seqlen > 0:
                    key_cache = key_cache[:, :, :trim_seqlen]
                    value_cache = value_cache[:, :, :trim_seqlen]
                else: # trim_seqlen == 0
                    key_cache = None
                    value_cache = None

            if reindex_batch_indices is not None:
                key_cache = Tensor(key_cache.data[reindex_batch_indices])
                value_cache = Tensor(value_cache.data[reindex_batch_indices])

            decoder_block.attn.kv_cache = (key_cache, value_cache)
