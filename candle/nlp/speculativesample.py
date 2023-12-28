"""Speculative sampling decoding.

References:
    [1] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper.
        Accelerating Large Language Model Decoding with Speculative Sampling
        arXiv:2302.01318, 2023.
    
"""
from __future__ import annotations

import numpy as np
import candle
import candle.functions as F
from candle.nlp.generation import top_k_sample, nucleus_sample, default_modify_kv_cache


def speculative_sample(target_model,
                       draft_model,
                       K: int,
                       indices: Tensor,
                       n_tokens_to_gen: int,
                       top_k: int = None,
                       top_p: float = None,
                       temperature: float = 1.0,
                       modify_kv_cache_func: Callable = None):
    """Generates response using speculative sampling and KV caching.
    
    Args:
        target_model, draft_model (model): Target and draft models. See speculative sample paper [1].
        K (int): Draft length. See speculative sample paper [1].
        indices (Tensor): Conditioning sequence with shape (1, seqlen).
        n_tokens_to_gen (int): Number of tokens to generate.
        top_k (int): Filter probabilities to those in the top k.
        top_p (float): Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
        temperature (float: Higher temperature raises the likelihood of lower probability sequences.
        modify_kv_cache_func (Callable): Callable with signature
            `modify_kv_cache_func(model, trim_seqlen, reindex_batch_indices),
            that modifies the model's kv_cache by trimming or reindexing.
            See `default_modify_kv_cache` for example implementation.
        
    Returns:
        generator. The generator will yield tokens as soon as they are sampled.

    """
    target_model.eval()
    draft_model.eval()
    
    def append(indices, new_index):
        return F.concat([indices, candle.Tensor([[new_index]])], axis=1)
    
    if modify_kv_cache_func is None: 
        modify_kv_cache_func = default_modify_kv_cache
    
    # Prefill KV cache
    _ = target_model(indices[:, :-1], use_kv_cache=True)
    _ = draft_model(indices[:, :-1], use_kv_cache=True)
    
    initial_seqlen = target_model.get_kv_cache_seqlen()
    assert initial_seqlen == draft_model.get_kv_cache_seqlen()
    
    token_n = 0
    while token_n < n_tokens_to_gen:
        # Sample K draft tokens auto-regressively 
        
        with candle.no_grad():
            draft_logits = []
            draft_indices = indices.clone()
            for _ in range(K):
                next_token_logits = draft_model(draft_indices[:, -1:], use_kv_cache=True)[:, -1]
                draft_logits.append(next_token_logits)
                
                probs = logits_to_probs(next_token_logits, temperature, top_k, top_p)[:, 0]
                next_index = np.random.choice(len(probs), p=probs)
                draft_indices = append(draft_indices, next_index)
        
            draft_logits = F.concat(draft_logits)
        
        # Compute K+1 sets of target logits
        
        target_logits = target_model(draft_indices[:, -(K + 1):], use_kv_cache=True)[0]
        
        # Accept/reject draft tokens based on rejection sampling scheme
        
        target_probs = logits_to_probs(target_logits, temperature, top_k, top_p)
        draft_probs = logits_to_probs(draft_logits, temperature, top_k, top_p)
        
        all_tokens_accepted = True
        cur_len = indices.shape[1]
        for t in range(K):
            q = target_probs[:, t]
            p = draft_probs[:, t]
            
            draft_idx = int(draft_indices[:, cur_len + t].data[0])
            
            if np.random.random() < q[draft_idx] / p[draft_idx]:  # Accept
                token_n += 1
                yield [draft_idx]
                indices = append(indices, draft_idx)
            
            else:  # Reject
                q_minus_p = (q - p).clip(min=0.0)
                next_index = np.random.choice(len(q), p=q_minus_p / q_minus_p.sum())
                token_n += 1
                yield [next_index]
                
                indices = append(indices, next_index)
                all_tokens_accepted = False
                break
    
        if all_tokens_accepted:
            next_index = np.random.choice(len(target_probs), p=target_probs[:, -1])
            token_n += 1
            yield [next_index]
            indices = append(indices, next_index)
        
        # Maintain KV cache
        modify_kv_cache_func(target_model, trim_seqlen=initial_seqlen + token_n)
        modify_kv_cache_func(draft_model, trim_seqlen=initial_seqlen + token_n)
    
    # We have to feed the last generated indices back in to maintain the KV cache
    _ = draft_model(indices[:, -1:], use_kv_cache=True)
    _ = target_model(indices[:, -1:], use_kv_cache=True)


def logits_to_probs(logits, temperature: float = 1.0, top_k: int = None, top_p: int = None):
    """
    Args:
        logits (Tensor): shape (N, vocab_size)

    Returns:
        probs (np.array): shape (vocab_size, N)

    """
    probs = F.softmax(logits / (temperature + 1e-6)).data.T
    
    if top_k is not None:
        probs = top_k_sample(probs, top_k)
    
    if top_p is not None:
        probs = nucleus_sample(probs, top_p)

    return probs
    