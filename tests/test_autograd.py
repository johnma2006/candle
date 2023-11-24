import numpy as np
from typing import List, Tuple
import unittest

import candle
import candle.functions as F
from candle import Tensor, Parameter
from .utils import model_numerical_grad_check


class TestAutograd(unittest.TestCase):
    
    def test_mlp(self):
        
        class MLP(candle.Module):

            def __init__(self,
                         input_size: int,
                         hidden_sizes: List[int]):
                super().__init__()
                self.linear_layers = candle.ParameterList([
                    candle.Linear(i, j)
                    for (i, j) in zip([input_size] + hidden_sizes, hidden_sizes)
                ])


            def forward(self, x):
                for linear_layer in self.linear_layers[:-1]:
                    x = linear_layer(x)
                    x = F.relu(x)

                x = self.linear_layers[-1](x)

                return x


        def loss_fn(model, random_seed):
            random_state = np.random.RandomState(random_seed)
            x = Tensor(random_state.normal(size=(12, 64)))
            y = Tensor((random_state.uniform(size=12) * 10).astype(int))

            output = model(x)
            loss = F.cross_entropy_loss(output, y)

            return loss

        # Switch to double precision for grad checks
        default_dtype = Tensor.DEFAULT_DTYPE
        Tensor.DEFAULT_DTYPE = np.float64

        model = MLP(input_size=64,
                    hidden_sizes=[128, 128, 128, 10])

        model_numerical_grad_check(model, loss_fn)
        
        Tensor.DEFAULT_DTYPE = default_dtype
        
        
    def test_resnet(self):
        
        class ResNetBlock(candle.Module):

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

                if in_channels != out_channels:
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


        class ResNet(candle.Module):

            def __init__(self,
                         num_classes: int,
                         in_channels: int,
                         resnet_blocks: List[Tuple[int, int, int]]):
                super().__init__()
                self.num_classes = num_classes

                self.conv = candle.Conv2d(in_channels, resnet_blocks[0][0], kernel_size=7, padding=3, stride=2)
                self.batch_norm = candle.BatchNorm(axis=(0, 2, 3))
                self.max_pool = candle.MaxPool2d(kernel_size=2)

                self.residual_blocks = candle.ParameterList([
                    ResNetBlock(in_channels, out_channels, stride)
                    for (in_channels, out_channels, stride) in resnet_blocks

                ])

                self.linear = candle.Linear(resnet_blocks[-1][1], num_classes)


            def forward(self, x):
                x = self.conv(x)
                x = self.batch_norm(x)
                x = F.relu(x)

                x = self.max_pool(x)

                for residual_block in self.residual_blocks:
                    x = residual_block(x)

                x = x.mean(axis=(2, 3))
                x = self.linear(x)

                return x


        def loss_fn(model, random_seed):
            random_state = np.random.RandomState(random_seed)
            x = Tensor(random_state.normal(size=(2, 3, 17, 19)))
            y = Tensor((random_state.uniform(size=2) * 10).astype(int))

            output = model(x)
            loss = F.cross_entropy_loss(output, y)

            return loss

        # Switch to double precision for grad checks
        default_dtype = Tensor.DEFAULT_DTYPE
        Tensor.DEFAULT_DTYPE = np.float64

        model = ResNet(num_classes=10,
                       in_channels=3,
                       resnet_blocks=[
                           # (in_channels, out_channels, stride)
                           (16, 16, 1),
                           (16, 32, 2),
                       ])

        model_numerical_grad_check(model, loss_fn)
        
        Tensor.DEFAULT_DTYPE = default_dtype
        
        
    def test_outdegree(self):
        # Checks that outdegree is 0 after backprop
        x1 = Parameter(Tensor(np.array([1, 2, 3])))
        x2 = x1[:]
        x3 = x2 + x1
        x4 = x3 ** 2
        x5 = x4.sum()
        
        x5._initialize_outdegree()
        
        assert x1._outdegree == 2
        assert x2._outdegree == 1
        assert x3._outdegree == 1
        assert x4._outdegree == 1
        assert x5._outdegree == 0

        x5.backward()
        
        assert x1._outdegree == 0
        assert x2._outdegree == 0
        assert x3._outdegree == 0
        assert x4._outdegree == 0
        assert x5._outdegree == 0
        
        
    def test_requires_grad_computation(self):
        a = Tensor(np.array([0, 1, 2, 3]))
        b = Parameter(Tensor(np.array([0, 1, 2, 3])))
        c = Tensor(np.array([0, 1, 2, 3]))

        assert a.requires_grad == False
        assert b.requires_grad == True
        assert c.requires_grad == False

        d = a + b
        e = c + d

        e._initialize_requires_grad_computation()
        assert a._requires_grad_computation == False
        assert b._requires_grad_computation == True
        assert c._requires_grad_computation == False
        assert d._requires_grad_computation == True
        assert e._requires_grad_computation == True
        
        
    def test_zero_grad(self):
        x = Tensor(np.random.random(size=(12, 128)))
        y = Tensor(np.random.random(size=(256)))

        layer = candle.Linear(128, 256)

        for i in range(1, 10):
            loss = (layer(x) - y).mean()
            loss.backward()
            assert np.isclose(i, (layer.W.grad / layer.W._batch_grad).min())
            assert np.isclose(i, (layer.W.grad / layer.W._batch_grad).max())

        layer.zero_grad()

        for i in range(1, 10):
            loss = (layer(x) - y).mean()
            loss.backward()
            assert np.isclose(i, (layer.W.grad / layer.W._batch_grad).min())
            assert np.isclose(i, (layer.W.grad / layer.W._batch_grad).max())
