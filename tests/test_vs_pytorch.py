import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle as candle
import candle.functions as F
import candle.optimizer


class TestEquivalencyVsPytorch(unittest.TestCase):
    
    def test_resnet_block_backprop_and_sgd(self):
        # ----------------------
        # Pytorch Implementation
        # ----------------------

        class ResNetBlock_Pytorch(nn.Module):

            def __init__(self,
                         in_channels: int,
                         out_channels: int,
                         stride: int = 1):
                super().__init__()

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.stride = stride

                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

                self.batch_norm1 = nn.BatchNorm2d(out_channels)
                self.batch_norm2 = nn.BatchNorm2d(out_channels)

                if in_channels != out_channels:
                    self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                else:
                    self.res_conv = None


            def forward(self, x):
                x_conv = self.conv1(x)
                x_conv = self.batch_norm1(x_conv)
                x_conv = torch.relu(x_conv)

                x_conv = self.conv2(x_conv)
                x_conv = self.batch_norm2(x_conv)

                if self.res_conv is not None:
                    x = self.res_conv(x)

                x_conv = x + x_conv
                x_conv = torch.relu(x_conv)

                return x_conv


        model_pytorch = ResNetBlock_Pytorch(8, 16, 2)

        x = np.random.normal(size=(16, 8, 35, 41))
        x_pytorch = torch.Tensor(x)
        _ = model_pytorch(x_pytorch)

        # -----------------------
        # candle Implementation
        # -----------------------

        class ResNetBlock_candle(candle.Module):

            def __init__(self,
                         in_channels: int,
                         out_channels: int,
                         stride: int = 1):
                super().__init__()

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.stride = stride

                self.conv1 = candle.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
                self.conv2 = candle.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

                self.batch_norm1 = candle.BatchNorm(axis=(0, 2, 3))
                self.batch_norm2 = candle.BatchNorm(axis=(0, 2, 3))

                if in_channels != out_channels or stride == 1:
                    self.res_conv = candle.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                else:
                    self.res_conv = None


            def forward(self, x):
                x_conv = self.conv1(x)
                x_conv = self.batch_norm1(x_conv)
                x_conv = F.relu(x_conv)

                x_conv = self.conv2(x_conv)
                x_conv = self.batch_norm2(x_conv)

                if self.res_conv is not None:
                    x = self.res_conv(x)

                x_conv = x + x_conv
                x_conv = F.relu(x_conv)

                return x_conv

        model_candle = ResNetBlock_candle(8, 16, 2)

        x_candle = candle.Tensor(x)
        _ = model_candle(x_candle)

        # ---------------------------------------------------
        # Transfer PyTorch model parameters to candle model
        # ---------------------------------------------------

        params_pytorch = dict(model_pytorch.named_parameters())
        params_candle = model_candle.parameters()

        for (key, value) in {
            'batch_norm1.beta': 'batch_norm1.bias',
            'batch_norm1.gamma': 'batch_norm1.weight',
            'batch_norm2.beta': 'batch_norm2.bias',
            'batch_norm2.gamma': 'batch_norm2.weight',
            'conv1.bias': 'conv1.bias',
            'conv2.bias': 'conv2.bias',
            'res_conv.bias': 'res_conv.bias',
        }.items():
            params_candle[key].data = (
                params_pytorch[value].detach().numpy().reshape(params_candle[key].shape).copy()
            )

        for (key, value) in {
            'conv1.kernel': 'conv1.weight',
            'conv2.kernel': 'conv2.weight',
            'res_conv.kernel': 'res_conv.weight',
        }.items():
            params_candle[key].data = (
                params_pytorch[value].swapaxes(0, 1).detach().numpy().reshape(params_candle[key].shape).copy()
            )

        # -----------------------
        # Test output equivalency
        # -----------------------

        output_pytorch = model_pytorch(x_pytorch)
        output_candle = model_candle(x_candle)

        ratio = output_pytorch.detach().numpy() / output_candle.data
        ratio[np.isnan(ratio)] = 1.0
        diff = output_pytorch.detach().numpy() - output_candle.data

        assert 1 - 5e-2 < ratio.min() < ratio.max() < 1 + 5e-2
        assert -1e-2 < diff.min() < diff.max() < 1e-2

        # ---------------------------------
        # Test Backprop and SGD equivalency
        # ---------------------------------

        loss_pytorch = output_pytorch.sum()
        loss_candle = output_candle.sum()

        loss_pytorch.backward()
        loss_candle.backward()

        optimizer_pytorch = torch.optim.SGD(model_pytorch.parameters(),
                                            lr=1e-1,
                                            weight_decay=1e-2)

        optimizer_candle = candle.optimizer.SGD(model_candle.parameters(),
                                                    learning_rate=1e-1,
                                                    weight_decay=1e-2)

        optimizer_pytorch.step()
        optimizer_candle.step()
        
        # Test gradient equivalency

        candle_name_to_pytorch_name= {
            'batch_norm1.beta': 'batch_norm1.bias',
            'batch_norm1.gamma': 'batch_norm1.weight',
            'batch_norm2.beta': 'batch_norm2.bias',
            'batch_norm2.gamma': 'batch_norm2.weight',
            'conv1.bias': 'conv1.bias',
            'conv1.kernel': 'conv1.weight',
            'conv2.bias': 'conv2.bias',
            'conv2.kernel': 'conv2.weight',
            'res_conv.bias': 'res_conv.bias',
            'res_conv.kernel': 'res_conv.weight',
        }

        for key in candle_name_to_pytorch_name:
            grad1 = params_pytorch[candle_name_to_pytorch_name[key]].detach().numpy()
            grad2 = params_candle[key].data
            assert 1 - 5e-2 < grad1.std() / grad2.std() < 1 + 5e-2
            assert 1 - 1e-2 < np.abs(grad1).mean() / np.abs(grad2).mean() < 1 + 1e-2

            
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
                                         num_heads=8,
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
        