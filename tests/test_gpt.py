import numpy as np
import unittest

import torch
from transformers import GPT2LMHeadModel

import candle
import candle.functions as F
from candle import Tensor
from candle.models.gpt.model import GPT, DecoderBlock


class TestGPT(unittest.TestCase):
    
    def test_gpt_use_kv_cache_vs_no_cache_equivalency(self):
        model = DecoderBlock(512, 8, 0.1)
        model.eval()

        x1 = Tensor(np.random.normal(size=(16, 10, 512)))
        x2 = Tensor(np.random.normal(size=(16, 10, 512)))
        kv_cache = None

        output1 = model(x1, use_kv_cache=True)
        output2 = model(x2, use_kv_cache=True)

        output_kv = F.concat([output1, output2], axis=1)
        output = model(F.concat([x1, x2], axis=1), use_kv_cache=False)

        ratio = (output_kv.data / output.data)

        assert 1 - 1e-3 < np.quantile(ratio, 0.01), np.quantile(ratio, 0.99) < 1 + 1e-3
        
        
    def test_decoder_use_kv_cache_vs_no_cache_equivalency(self):
        model = GPT(
            num_layers=12,
            num_heads=8,
            embed_dim=32,
            vocab_size=200,
            block_size=128,
            dropout_p=0.1,
        )
        model.eval()

        # With KV Caching

        indices_cumulative = []
        logits_cumulative = []

        for _ in range(60):
            indices = Tensor((np.random.random(size=(16, 1)) * 10).astype(int))    
            logits = model(indices, use_kv_cache=True)

            indices_cumulative.append(indices)
            logits_cumulative.append(logits)

        logits_kv_cache = F.concat(logits_cumulative, axis=1)
        indices_cumulative = F.concat(indices_cumulative, axis=1)

        # Without KV Caching

        logits = model(indices_cumulative, use_kv_cache=False)

        ratio = logits.data / logits_kv_cache.data
        assert 1 - 1e-3 < np.quantile(ratio, 0.01) < np.quantile(ratio, 0.99) < 1 + 1e-3
