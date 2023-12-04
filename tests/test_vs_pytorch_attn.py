import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle


class TestAttnModulesVsPytorch(unittest.TestCase):
            
    def test_dot_product_attn_w_mask(self):
        # Candle implementation

        query = candle.Tensor(np.random.normal(size=(13, 14, 32, 512)))
        key = candle.Tensor(np.random.normal(size=(13, 14, 42, 512)))
        value = candle.Tensor(np.random.normal(size=(13, 14, 42, 512)))
        attn_mask = candle.Tensor((np.random.random(size=(32, 42)) > 0.5).astype(int))

        self = candle.DotProductAttention(0.0)

        (attn_output, attn_scores) = self(query, key, value, attn_mask)

        # PyTorch implementation

        query_torch = torch.Tensor(query.data)
        key_torch = torch.Tensor(key.data)
        value_torch = torch.Tensor(value.data)
        attn_mask_torch = torch.Tensor(1 - attn_mask.data).bool()

        attn_output_torch = torch.nn.functional.scaled_dot_product_attention(query_torch, key_torch, value_torch,
                                                                             attn_mask_torch, dropout_p=0)

        diff = attn_output.data - attn_output_torch.numpy()

        assert -1e-5 < diff.min(), diff.max() < 1e-5
        
        
    def test_dot_product_attn(self):
        # Candle implementation

        query = candle.Tensor(np.random.normal(size=(13, 14, 32, 512)))
        key = candle.Tensor(np.random.normal(size=(13, 14, 42, 512)))
        value = candle.Tensor(np.random.normal(size=(13, 14, 42, 512)))
        attn_mask = None

        self = candle.DotProductAttention(0.0)

        (attn_output, attn_scores) = self(query, key, value, attn_mask)

        # PyTorch implementation

        query_torch = torch.Tensor(query.data)
        key_torch = torch.Tensor(key.data)
        value_torch = torch.Tensor(value.data)
        attn_mask_torch = None

        attn_output_torch = torch.nn.functional.scaled_dot_product_attention(query_torch, key_torch, value_torch,
                                                                             attn_mask_torch, dropout_p=0)

        diff = attn_output.data - attn_output_torch.numpy()

        assert -1e-5 < diff.min(), diff.max() < 1e-5
        
        
    def test_multihead_attn(self):
        # Candle implementation

        self = candle.MultiheadAttention(embed_dim=512,
                                         n_heads=8,
                                         dropout_p=0.0)

        # Initialize PyTorch attention and transfer parameters

        self_pytorch = nn.MultiheadAttention(embed_dim=512,
                                             num_heads=8,
                                             dropout=0.0,
                                             batch_first=True)

        params = dict(self_pytorch.named_parameters())

        for (i, param) in zip(range(3), [self.W_q, self.W_k, self.W_v]):
            param.W.data[:] = params['in_proj_weight'].detach().numpy()[i*self.embed_dim:(i+1)*self.embed_dim, :].copy().T
            param.b.data[:] = params['in_proj_bias'].detach().numpy()[i*self.embed_dim:(i+1)*self.embed_dim].copy()

        self.W_o.W.data[:] = params['out_proj.weight'].detach().numpy().copy().T
        self.W_o.b.data[:] = params['out_proj.bias'].detach().numpy().copy()

        # Feed key, query, value and match outputs

        key = candle.Tensor(np.random.normal(size=(13, 32, 512)))
        query = candle.Tensor(np.random.normal(size=(13, 32, 512)))
        value = candle.Tensor(np.random.normal(size=(13, 32, 512)))
        attn_mask = candle.Tensor((np.random.random(size=(32, 32)) > 0.5).astype(int))

        key_pytorch = torch.Tensor(key.data)
        query_pytorch = torch.Tensor(query.data)
        value_pytorch = torch.Tensor(value.data)
        attn_mask_pytorch = torch.Tensor(attn_mask.data).bool()

        (attn_output, attn_scores) = self(key, query, value, attn_mask=attn_mask)
        (attn_output_pytorch, attn_scores_pytorch) = self_pytorch(key_pytorch, query_pytorch, value_pytorch, attn_mask=attn_mask_pytorch)

        diff = (attn_output.data - attn_output_pytorch.detach().numpy()).flatten()
        ratio = (attn_output.data / attn_output_pytorch.detach().numpy()).flatten()

        assert -1e-5 < min(diff) < max(diff) < 1e-5
        assert 1 - 1e-5 < np.quantile(ratio, 0.05) < np.quantile(ratio, 0.95) < 1 + 1e-5
        