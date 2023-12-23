import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle


class TestBatchNormVsPytorch(unittest.TestCase):
            
    def test_batch_norm(self):
        num_features = 100
        bn1 = candle.BatchNorm(num_features)
        bn2 = nn.BatchNorm2d(num_features)

        # Test batch norm under model.eval()

        bn1.eval()
        bn2.eval()

        x = candle.Tensor(np.random.normal(size=(16, num_features, 32, 32)))
        x_torch = torch.Tensor(x.data)

        assert np.isclose(float(bn1(x).sum().data), bn2(x_torch).sum().item(), atol=0.1)

        # Test batch norm in model.train()

        bn1.train()
        bn2.train()

        # Keep feeding in data to update running mean and running_var

        for _ in range(10):
            x = candle.Tensor(np.random.normal(size=(16, num_features, 32, 32)))
            x_torch = torch.Tensor(x.data)

            # Asssert 
            assert np.isclose(float(bn1(x).sum().data), bn2(x_torch).sum().item(), atol=0.01)

            # Assert running_mean and running_var are the same
            assert np.isclose(bn1.ema_mean.flatten().sum(), bn2.running_mean.sum().item(), atol=0.01)
            assert np.isclose(bn1.ema_var.flatten().sum(), bn2.running_var.sum().item(), atol=0.01)

        # Test batch norm again under model.eval()

        bn1.eval()
        bn2.eval()

        x = candle.Tensor(np.random.normal(size=(16, num_features, 32, 32)))
        x_torch = torch.Tensor(x.data)