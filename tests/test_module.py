import numpy as np
from typing import List, Tuple
import unittest

import candle
import candle.functions as F
from candle import Tensor, Parameter
from .utils import model_numerical_grad_check


class TestModule(unittest.TestCase):
    
    def test_module_parameters_under_weight_tying(self):
        
        class TestModule(candle.Module):

            def __init__(self):
                super().__init__()
                self.linear = candle.Linear(20, 30)

                self.Ws = candle.ParameterList([
                    self.linear.W,
                    self.linear.W,
                    self.linear.W,
                    self.linear.W,
                    self.linear.W,
                ])

            def forward(self):
                return 0

        model = TestModule()
        assert len(model.parameters()) == 2

        param = Parameter(Tensor([1, 2, 3]))
        Ws = candle.ParameterList([param, param, param])
        assert len(Ws.parameters()) == 1

        linear = candle.Linear(20, 30)
        Ws = candle.ParameterList([linear.W, linear.W, linear.W, linear.W, linear.W])
        assert len(Ws.parameters()) == 1
        