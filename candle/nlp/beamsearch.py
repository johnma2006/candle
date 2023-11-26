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
                        sample: bool = True):
    """Given a conditioning sequence, generates N more tokens using beam search.

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
        Nucleus sampling. Filter to top probs such that the sum prob is just less than top_p.

        References:
        [1] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.
            The Curious Case of Neural Text Degeneration. arXiv:1904.09751, 2019
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities, False to take argmax.
        
    Returns
    -------
    generator
        The generator will yield tokens greedily as soon as all current `beam_size`
        candidates agree on that token.

    """
    model.eval()
    
    if beam_size is None:
        beam_size = 1

    # We will maintain `beam_search_cumulative_log_prob` to have the same length as `indices`
    # On the 0th iter, `indices` has shape (1, initial_seqlen).
    # Afterwards, `indices` has shape (beam_size, initial_seqlen + iter).
    beam_search_cumulative_log_prob = np.array([0.0])
    indices = Tensor(indices.data[None, :])

    # indices[:, head_index] is the first token that hasn't been yielded yet
    head_index = indices.shape[1]

    for _ in range(n_tokens_to_generate):
        # If indices is too long, filter to last `block_size` tokens
        
        with candle.no_grad():
            if indices.shape[1] <= model.block_size:
                logits = model(indices)
            else:
                logits = model(indices[:, -model.block_size])

        probs = F.softmax(logits[:, -1] / temperature).data.T  # shape (vocab_size, beam_size)

        if top_k is not None:
            probs = top_k_sample(probs, top_k)

        if top_p is not None:
            probs = nucleus_sample(probs, top_p)

        # Accumulate prob by scaling by `beam_search_cumulative_log_prob`
        probs += 1.0e-3 / len(probs)  # Add small eps prob to guarentee that num_non_zero(probs) >= beam_size
        log_probs = np.log(probs) + beam_search_cumulative_log_prob

        # `next_beam_indices` are the best `beam_size` candidates
        # Each `beam_index` in `next_beam_indices` is defined as:
        #    beam_index // indices.shape[0] is the index of the next token
        #    indices[beam_index % indices.shape[0]] is which that token will be appended to
        if sample:
            normalized_probs = candle.utils.softmax(log_probs.flatten())
            next_beam_indices = np.random.choice(np.arange(log_probs.size),
                                                 size=beam_size,
                                                 p=normalized_probs,
                                                 replace=False)
        else:
            next_beam_indices = log_probs.flatten().argsort()[-beam_size:]

        # Update beam_search_cumulative_log_prob
        beam_search_cumulative_log_prob = log_probs.flatten()[next_beam_indices]

        # Update indices by appending the tokens from next_beam_indices
        new_indices = np.zeros((beam_size, indices.shape[1] + 1)).astype(int)
        for (i, beam_index) in enumerate(next_beam_indices):
            next_index = beam_index // indices.shape[0]
            indices_i = beam_index % indices.shape[0]

            new_indices[i] = np.append(indices[indices_i].data, next_index)
        indices = Tensor(new_indices)

        # If all `beam_index` candidates have the same token at the head, then it's
        # safe to yield that token; then, increment the head by 1
        indices_at_head = indices.data[:, head_index]
        while np.unique(indices_at_head).size == 1:
            yield np.array([indices_at_head[0]])
            head_index += 1
            
            if head_index == indices.shape[1]:
                break
            
            indices_at_head = indices.data[:, head_index]

    # Yield rest of the generated indices

    if sample:
        best_index = np.random.choice(np.arange(indices.shape[0]),
                                      p=candle.utils.softmax(beam_search_cumulative_log_prob))
    else:
        best_index = beam_search_cumulative_log_prob.argmax()

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
