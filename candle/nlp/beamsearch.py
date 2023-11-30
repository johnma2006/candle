"""Beam search decoding algorithm."""

import numpy as np

from ..tensor import Tensor
import candle
import candle.functions as F


def beam_search_decoder(model, 
                        indices: Tensor,
                        n_tokens_to_generate: int,
                        beam_size: int,
                        top_k: int = None,
                        top_p: float = None,
                        temperature: float = 1.0,
                        sample: bool = True,
                        use_kv_cache: bool = True):
    """Generates tokens using beam search with KV caching.
    
    Tokens will be yielded as soon as all `beam_size` beams agree on that token.
    
    If KV caching, we will guarantee that:
        (kv_cache_seqlen after decoding) == (kv_cache_seqlen before decoding) + (num tokens yielded)
        - This will be true even if the generator terminates early, as long as generator.close() is called
        - Note this means that we feed in the last generated token to update the KV cache

    Parameters
    ----------
    model
        Model instance. model should take in tensors of shape (batch, seqlen,) and output logits of
        shape (batch, seqlen, vocab_size).
    indices
        Conditioning sequence. Tensor of shape (seqlen,).
    n_tokens_to_generate
        Number of tokens to generate.
    beam_size
        Beam size to use in beam search. e.g., beam_size = 1 for greedy search.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum prob is less than top_p.
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities, False to take argmax.
    use_kv_cache
        If True, uses kv_cache to speed up inference. If generator terminates early, make sure to call
        generator.close() to properly maintain the KV cache.
        
        If using KV caching, then model must implement
            model.modify_kv_cache(trim_seqlen, reindex_batch_indices)
        
    Returns
    -------
    generator
        The generator will yield tokens as soon as all `beam_size` beams agree on that token.
        
    Examples
    --------
    (Ppseudocode) In the following, response2 == kv_response2 while being faster to generate.

        prompt1 = 'Hi, I am John.'
        prompt2 = 'That is great.'

        response1 = beam_search_decoder(prompt1, use_kv_cache=False)
        response2 = beam_search_decoder(prompt1 + response1 + prompt2, use_kv_cache=False)

        kv_response1 = beam_search_decoder(prompt1, use_kv_cache=True)
        kv_response2 = beam_search_decoder(prompt2, use_kv_cache=True)

    """
    model.eval()
    
    indices_seqlen = len(indices)

    if beam_size is None:
        beam_size = 1

    # We maintain `cumulative_log_prob_per_beam` to have the same length as `indices`
    # On the 0th iter, `indices` has shape (1, initial_seqlen).
    # Afterwards, `indices` has shape (beam_size, initial_seqlen + iter).
    cumulative_log_prob_per_beam = np.array([0.0])
    indices = Tensor(indices.data[None, :])

    # indices[:, head_index] is the first token that hasn't been yielded yet
    head_index = indices.shape[1]

    for token_n in range(n_tokens_to_generate):
        # If indices is too long, filter to last `block_size` tokens
        indices_to_input = indices[:, -model.block_size:]

        with candle.no_grad():
            if use_kv_cache:
                # After the first step, feed in one token at a time
                if token_n > 0:
                    indices_to_input = indices_to_input[:, -1:]

                # Trim kv_cache in case it gets too long
                # TODO: KV cache won't be perfectly maintained right now if this happens
                model.modify_kv_cache(trim_seqlen=model.block_size - indices_to_input.shape[1])

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

        # Update cumulative_log_prob_per_beam
        cumulative_log_prob_per_beam = log_probs.flatten()[next_beam_indices]

        # Update indices by appending the tokens from next_beam_indices
        new_indices = np.zeros((beam_size, indices.shape[1] + 1)).astype(int)
        reindex_kv_indices = []
        for (i, beam_index) in enumerate(next_beam_indices):
            next_index = beam_index // indices.shape[0]
            indices_i = beam_index % indices.shape[0]

            new_indices[i] = np.append(indices[indices_i].data, next_index)
            reindex_kv_indices.append(indices_i)
        indices = Tensor(new_indices)

        # Realign kv_cache with new beam search indices
        if use_kv_cache:
            model.modify_kv_cache(reindex_batch_indices=reindex_kv_indices)

        # If all `beam_index` beams have the same token at head_index, then it's
        # safe to yield that token
        indices_at_head = indices.data[:, head_index]
        while np.unique(indices_at_head).size == 1:
            try:
                head_index += 1
                yield np.array([indices_at_head[0]])

            except GeneratorExit:
                # This means that the generator exited early. We have to do some cleanup to
                # maintain the KV cache.
                if use_kv_cache:
                    # Update the kv_cache with the last generated token.
                    model.modify_kv_cache(trim_seqlen=model.block_size - 1)
                    _ = model(indices[:, -1:], use_kv_cache=True)

                    # Filter KV cache to first `head_index` tokens, and take arbitrary beam (they're
                    # all the same)
                    model.modify_kv_cache(trim_seqlen=-head_index,
                                          reindex_batch_indices=[0])
                return

            indices_at_head = indices.data[:, head_index]
            if head_index == indices.shape[1]:
                break

    # Yield rest of the generated indices
    if sample:
        best_index = np.random.choice(np.arange(indices.shape[0]),
                                      p=candle.utils.softmax(cumulative_log_prob_per_beam))
    else:
        best_index = cumulative_log_prob_per_beam.argmax()

    # To guarantee that:
    #     (kv_cache_seqlen after decoding) == (kv_cache_seqlen before decoding) + (num tokens yielded)
    # we update the kv_cache with the last generated token.
    if use_kv_cache:
        model.modify_kv_cache(trim_seqlen=model.block_size - 1,
                              reindex_batch_indices=[best_index])
        _ = model(indices[[best_index], -1:], use_kv_cache=True)
        
    yield indices.data[best_index, head_index:]
    
    
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



