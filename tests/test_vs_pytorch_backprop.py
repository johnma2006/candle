import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle
import candle.functions as F
import candle.optimizer


class TestBackpropVsPytorch(unittest.TestCase):
    
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
        model_candle.eval()
        x_candle = candle.Tensor(x)
        _ = model_candle(x_candle)  # Initialize batchnorm params
        model_candle.train()

        # ---------------------------------------------------
        # Transfer PyTorch model parameters to candle model
        # ---------------------------------------------------

        params_pytorch = dict(model_pytorch.named_parameters())
        params_candle = model_candle.parameters()

        for (key, value) in {
            'batch_norm1.W': 'batch_norm1.weight',
            'batch_norm1.b': 'batch_norm1.bias',
            'batch_norm2.W': 'batch_norm2.weight',
            'batch_norm2.b': 'batch_norm2.bias',
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

        # ---------------------------------------
        # Test output equivalency - model.train()
        # ---------------------------------------
        
        model_pytorch.train()
        model_candle.train()

        output_pytorch = model_pytorch(x_pytorch)
        output_candle = model_candle(x_candle)

        ratio = output_pytorch.detach().numpy() / output_candle.data
        ratio[np.isnan(ratio)] = 1.0
        diff = output_pytorch.detach().numpy() - output_candle.data

        assert 1 - 1e-1 < ratio.min() < ratio.max() < 1 + 1e-1
        assert -1e-2 < diff.min() < diff.max() < 1e-2
        
        # --------------------------------------
        # Test output equivalency - model.eval()
        # --------------------------------------

        model_pytorch.eval()
        model_candle.eval()
        
        output_pytorch = model_pytorch(x_pytorch)
        output_candle = model_candle(x_candle)

        ratio = output_pytorch.detach().numpy() / output_candle.data
        ratio[np.isnan(ratio)] = 1.0
        diff = output_pytorch.detach().numpy() - output_candle.data

        assert -0.01 < diff.min() < diff.max() < 0.01

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
            'batch_norm1.W': 'batch_norm1.weight',
            'batch_norm1.b': 'batch_norm1.bias',
            'batch_norm2.W': 'batch_norm2.weight',
            'batch_norm2.b': 'batch_norm2.bias',
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

    