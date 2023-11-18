"""Arithmetic operations."""

import numpy as np
from typing import List

from .operation import Operation
from .. import utils
from ..tensor import Tensor


class Addition(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data + b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Subtraction(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data - b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(-output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
        

class Multiplication(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data * b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        output_grad_a = output_grad * b.data
        output_grad_b = output_grad * a.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
        
    
class Division(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data / b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
    
        (a, b) = self.inputs
        
        output_grad_a = output_grad / b.data
        output_grad_b = -output_grad * a.data / b.data / b.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Power(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 power: float):
        super().__init__(inputs)
        self.power = power
        
    
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(self.inputs[0].data ** self.power)
    
    
    def _backward(self,
                  output_grad: np.array):
        
        input_grad = output_grad * self.power * self.inputs[0].data ** (self.power - 1)
        return (input_grad,)