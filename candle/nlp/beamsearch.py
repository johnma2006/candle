"""Beam search decoding algorithm."""

import numpy as np

from ..tensor import Tensor
import candle
import candle.functions as F


def beam_search_generation(model, 
                           indices: Tensor,
                           n_tokens_to_generate: int,
                           beam_size: int,
                           top_k: int = None,
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
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences, False to take argmax.
        
    Returns
    -------
    generator
        The generator will yield tokens greedily as soon as all current `beam_size`
        candidates agree on that token.

    """
    model.eval()

    # We will maintain `beam_search_cumulative_log_prob` to have the same length as `indices`
    # On the 0th iter, `indices` has shape (1, initial_seqlen).
    # Afterwards, `indices` has shape (beam_size, initial_seqlen + iter).
    beam_search_cumulative_log_prob = np.array([0.0])
    indices = Tensor(indices.data[None, :])

    # indices[:, head_index] is the first token that hasn't been yielded yet
    head_index = indices.shape[1]

    for _ in range(n_tokens_to_generate):
        # If indices is too long, filter to last `block_size` tokens
        if indices.shape[1] <= model.block_size:
            logits = model(indices)
        else:
            logits = model(indices[:, -model.block_size])

        probs = F.softmax(logits[:, -1] / temperature).data.T  # shape (vocab_size, beam_size)

        if top_k is not None:
            # For each row, zero out everything except for top_k probs per row
            top_k_prob = np.sort(probs, axis=0)[-top_k, :]
            probs[probs < top_k_prob] = 0
            probs /= probs.sum(axis=0)

        # Accumulate prob by scaling by `beam_search_cumulative_log_prob`
        with np.errstate(divide='ignore'):
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
    