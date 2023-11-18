import numpy as np

from .operation import Operation
from .. import utils
from ..tensor import Tensor

    
class CrossEntropyLossOperation(Operation):
    
    def _forward(self):
        (logits, target) = self.inputs
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert len(target) == len(logits)
        
        log_softmax = utils.log_softmax(logits.data)

        return Tensor(-np.mean(log_softmax[range(len(target)), target.data]))
    
    
    def _backward(self,
                  output_grad: np.array):
        (logits, target) = self.inputs
        
        softmax = utils.softmax(logits.data)
        softmax[range(len(target)), target.data] -= 1
        
        return (output_grad * softmax / len(logits), np.zeros(len(target)))  # target has no gradient