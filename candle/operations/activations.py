"""Operations in a computation graph."""

import numpy as np

from .operation import Operation
from .. import tensor


class ReLUActivation(Operation):
    
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(np.maximum(self.inputs[0].data, 0.0))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad * (self.inputs[0].data > 0)
        
        return (input_grad,)
    
    
class GeLUActivation(Operation):
    """Gaussian Error Linear Unit activation function.

    References:
    [1] Dan Hendrycks, Kevin Gimpel.
        Gaussian Error Linear Units (GELUS). arXiv:1606.08415, 2016
        
    """
    
    def _forward(self):
        assert len(self.inputs) == 1
        x = self.inputs[0].data

        return tensor.Tensor(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))))
    
    
    def _backward(self,
                  output_grad: np.array):
        
        x = self.inputs[0].data
        
        tanh = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
        partial_grad = 0.5 * (1 + tanh + x * (1 - tanh ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2))
        
        input_grad = output_grad * partial_grad
        
        return (input_grad,)
    
    
class SiLUActivation(Operation):
    """Sigmoid Linear Units activation function.

    References:
    [1] Stefan Elfwing, Eiji Uchibe, Kenji Doya,
        Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning
        arXiv:1702.03118v3, 2017.
    
    """
    
    def _forward(self):
        assert len(self.inputs) == 1
        x = self.inputs[0].data
        sigmoid_x = 1 / (1 + np.exp(-x))

        return tensor.Tensor(sigmoid_x * x)
    
    
    def _backward(self,
                  output_grad: np.array):
        x = self.inputs[0].data
        sigmoid_x = 1 / (1 + np.exp(-x))
        partial_grad = sigmoid_x * (1 - sigmoid_x) * x + sigmoid_x
        input_grad = output_grad * partial_grad

        return (input_grad,)
    