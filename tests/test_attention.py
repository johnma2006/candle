import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle
import candle.functions as F
from candle import MultiheadAttention, GroupedQueryRotaryAttention, Tensor


class TestMultiHeadAttention(unittest.TestCase):

    def test_kv_cache_vs_no_cache_with_self_attention(self):
        # ----------------------------------------------------------------
        # Test self-attention equivalency of using kv_cache vs no-kv_cache
        # when using causal atten
        # ----------------------------------------------------------------

        self = MultiheadAttention(embed_dim=512, num_heads=8, dropout_p=0.1)
        self.eval()

        # with kv_cache

        xs = []

        attn_outputs = []
        for _ in range(10):
            x = Tensor(np.random.normal(size=(16, 3, 512)))
            causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))
            (attn_output, attn_scores) = self.forward(x, x, x, causal_attn_mask, use_kv_cache=True)

            xs.append(x)
            attn_outputs.append(attn_output)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)

        # without kv_cache

        x = F.concat(xs, axis=1)
        causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))

        (attn_output, attn_scores) = self.forward(x, x, x, causal_attn_mask, use_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-3 < np.quantile(ratio, 0.01) <= np.quantile(ratio, 0.99) < 1 + 1e-3

        
    def test_kv_cache_vs_no_cache_with_regular_attention(self):
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

        attn_outputs = []
        for _ in range(10):
            query = Tensor(np.random.normal(size=(16, 3, 512)))
            key = Tensor(np.random.normal(size=(16, 3, 512)))
            value = Tensor(np.random.normal(size=(16, 3, 512)))
            causal_attn_mask = Tensor(1 - np.tri(query.shape[1]))

            (attn_output, attn_scores) = self.forward(query, key, value, causal_attn_mask, use_kv_cache=True)

            attn_outputs.append(attn_output)
            queries.append(query)
            keys.append(key)
            values.append(value)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)

        # without kv_cache

        query = F.concat(queries, axis=1)
        key = F.concat(keys, axis=1)
        value = F.concat(values, axis=1)
        causal_attn_mask = Tensor(1 - np.tri(query.shape[1]))

        (attn_output, attn_scores) = self.forward(query, key, value, causal_attn_mask, use_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-3 < np.quantile(ratio, 0.005) < np.quantile(ratio, 0.995) < 1 + 1e-3

        
class TestGroupedQueryRotaryAttention(unittest.TestCase):

    def test_kv_cache_vs_no_cache_with_self_attention(self):
        # ----------------------------------------------------------------
        # Test self-attention equivalency of using kv_cache vs no-kv_cache
        # when using causal atten
        # ----------------------------------------------------------------

        self = GroupedQueryRotaryAttention(embed_dim=512, num_heads=8, dropout_p=0.1,
                                           num_groups=2, apply_rotary_embedding=True)
        self.eval()

        # with kv_cache

        xs = []

        attn_outputs = []
        for _ in range(10):
            x = Tensor(np.random.normal(size=(16, 3, 512)))
            causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))
            (attn_output, attn_scores) = self.forward(x, x, x, causal_attn_mask, use_kv_cache=True)

            xs.append(x)
            attn_outputs.append(attn_output)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)

        # without kv_cache

        x = F.concat(xs, axis=1)
        causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))

        (attn_output, attn_scores) = self.forward(x, x, x, causal_attn_mask, use_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-3 < np.quantile(ratio, 0.01) <= np.quantile(ratio, 0.99) < 1 + 1e-3

        
    def test_kv_cache_vs_no_cache_with_regular_attention(self):
        # ---------------------------------------------------------
        # Test equivalency of using kv_cache and not using kv_cache
        # when using causal atten
        # ---------------------------------------------------------

        self = GroupedQueryRotaryAttention(embed_dim=512, num_heads=8, dropout_p=0.1,
                                           num_groups=2, apply_rotary_embedding=True)
        self.eval()

        # with kv_cache

        queries = []
        keys = []
        values = []

        attn_outputs = []
        for _ in range(10):
            query = Tensor(np.random.normal(size=(16, 3, 512)))
            key = Tensor(np.random.normal(size=(16, 3, 512)))
            value = Tensor(np.random.normal(size=(16, 3, 512)))
            causal_attn_mask = Tensor(1 - np.tri(query.shape[1]))

            (attn_output, attn_scores) = self.forward(query, key, value, causal_attn_mask, use_kv_cache=True)

            attn_outputs.append(attn_output)
            queries.append(query)
            keys.append(key)
            values.append(value)

        attn_output_kv_caching = F.concat(attn_outputs, axis=1)

        # without kv_cache

        query = F.concat(queries, axis=1)
        key = F.concat(keys, axis=1)
        value = F.concat(values, axis=1)
        causal_attn_mask = Tensor(1 - np.tri(query.shape[1]))

        (attn_output, attn_scores) = self.forward(query, key, value, causal_attn_mask, use_kv_cache=False)

        ratio = attn_output_kv_caching.data / attn_output.data
        ratio[np.isnan(ratio)] = 1.0 

        assert 1 - 1e-3 < np.quantile(ratio, 0.005) < np.quantile(ratio, 0.995) < 1 + 1e-3
