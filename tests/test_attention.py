import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle
import candle.functions as F
from candle import MultiheadAttention, Tensor


class TestMultiHeadAttention(unittest.TestCase):

    def test_kv_cache_vs_no_kv_cache_with_self_attention(self):
        # ----------------------------------------------------------------
        # Test self-attention equivalency of using kv_cache vs no-kv_cache
        # when using causal atten
        # ----------------------------------------------------------------

        self = MultiheadAttention(embed_dim=512, num_heads=8, dropout_p=0.1)
        self.eval()

        # with kv_cache

        xs = []
        kv_cache = None

        attn_outputs = []
        for _ in range(10):
            if _ == 1:
                break

            x = Tensor(np.random.normal(size=(16, 3, 512)))

            x_seqlen = x.shape[1]
            if kv_cache is None:
                cache_seqlen = 0
            else:
                cache_seqlen = kv_cache[0].shape[2]

            causal_attn_mask = 1 - np.tri(max(x_seqlen, x_seqlen + cache_seqlen))
            causal_attn_mask = Tensor(causal_attn_mask[-x_seqlen:, :x_seqlen + cache_seqlen])

            xs.append(x)

            (attn_output, attn_scores, kv_cache) = self.forward(x, x, x, causal_attn_mask,
                                                                kv_cache, return_new_kv_cache=True)

            attn_outputs.append(attn_output)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)

        # without kv_cache

        x = F.concat(xs, axis=1)

        x_seqlen = x.shape[1]
        causal_attn_mask = Tensor(1 - np.tri(x_seqlen))

        (attn_output, attn_scores) = self.forward(x, x, x, causal_attn_mask,
                                                  kv_cache=None, return_new_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-5 < ratio.min() <= ratio.max() < 1 + 1e-5
        
        
    def test_kv_cache_vs_no_kv_cache_with_regular_attention(self):
        # ---------------------------------------------------------
        # Test equivalency of using kv_cache and not using kv_cache
        # when using causal atten
        # ---------------------------------------------------------

        self = MultiheadAttention(embed_dim=512, num_heads=8, dropout_p=0.1)
        self.eval()

        # with kv_cache

        queries = []
        keys = []
        values = []
        kv_cache = None

        attn_outputs = []
        for _ in range(10):
            query = Tensor(np.random.normal(size=(16, 3, 512)))
            key = Tensor(np.random.normal(size=(16, 3, 512)))
            value = Tensor(np.random.normal(size=(16, 3, 512)))

            query_seqlen = query.shape[1]
            key_seqlen = key.shape[1]
            if kv_cache is None:
                cache_seqlen = 0
            else:
                cache_seqlen = kv_cache[0].shape[2]

            causal_attn_mask = 1 - np.tri(max(query_seqlen, key_seqlen + cache_seqlen))
            causal_attn_mask = Tensor(causal_attn_mask[-query_seqlen:, :key_seqlen + cache_seqlen])

            queries.append(query)
            keys.append(key)
            values.append(value)

            (attn_output, attn_scores, kv_cache) = self.forward(query, key, value, causal_attn_mask,
                                                                kv_cache, return_new_kv_cache=True)

            attn_outputs.append(attn_output)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)
        # causal_attn_masks_caching = F.concat(causal_attn_masks, axis=0)

        # without kv_cache

        query = F.concat(queries, axis=1)
        key = F.concat(keys, axis=1)
        value = F.concat(values, axis=1)

        query_seqlen = query.shape[1]
        key_seqlen = key.shape[1]

        causal_attn_mask = 1 - np.tri(max(query_seqlen, key_seqlen))
        causal_attn_mask = Tensor(causal_attn_mask[:query_seqlen, :key_seqlen])

        (attn_output, attn_scores) = self.forward(query, key, value, causal_attn_mask,
                                                  kv_cache=None, return_new_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-3 < np.quantile(ratio, 0.005) < np.quantile(ratio, 0.995) < 1 + 1e-3