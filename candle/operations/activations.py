"""Operations in a computation graph."""

import numpy as np

from .operation import Operation
from ..tensor import Tensor


class ReLUActivation(Operation):
    
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(np.maximum(self.inputs[0].data, 0.0))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad * (self.inputs[0].data > 0)
        
        return (input_grad,)
    