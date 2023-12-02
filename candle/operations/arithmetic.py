"""Arithmetic operations."""

from __future__ import annotations
import numpy as np
from typing import List

from .operation import Operation
from .. import tensor, utils


class Addition(Operation):
    """f(inputs) = inputs[0] + inputs[1], with broadcasting"""
    
    def _forward(self):
        (a, b) = self.inputs
        return tensor.Tensor(a.data + b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Subtraction(Operation):
    """f(inputs) = inputs[0] - inputs[1], with broadcasting"""
    
    def _forward(self):
        (a, b) = self.inputs
        return tensor.Tensor(a.data - b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(-output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
        

class Multiplication(Operation):
    """f(inputs) = inputs[0] * inputs[1], with broadcasting"""
    
    def _forward(self):
        (a, b) = self.inputs
        return tensor.Tensor(a.data * b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        output_grad_a = output_grad * b.data
        output_grad_b = output_grad * a.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
        
    
class Division(Operation):
    """f(inputs) = inputs[0] / inputs[1], with broadcasting"""
    
    def _forward(self):
        (a, b) = self.inputs
        return tensor.Tensor(a.data / b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
    
        (a, b) = self.inputs
        
        output_grad_a = output_grad / b.data
        output_grad_b = -output_grad * a.data / b.data / b.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Power(Operation):
    """f(inputs) = inputs[0] ** power"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 power: float):
        super().__init__(inputs)
        self.power = power
        
    
    def _forward(self):
        assert len(self.inputs) == 1
        assert isinstance(self.power, (int, float, complex))
        return tensor.Tensor(self.inputs[0].data ** self.power)
    
    
    def _backward(self,
                  output_grad: np.array):
        
        input_grad = output_grad * self.power * self.inputs[0].data ** (self.power - 1)
        return (input_grad,)
    
    
class Exponentiation(Operation):
    """f(inputs) = power ** inputs[0]"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 base: float):
        super().__init__(inputs)
        self.base = base
        
    
    def _forward(self):
        assert len(self.inputs) == 1
        assert isinstance(self.base, (int, float, complex)) or np.issubdtype(self.base, np.number)
        return tensor.Tensor(self.base ** self.inputs[0].data)
    
    
    def _backward(self,
                  output_grad: np.array):
        
        input_grad = output_grad * np.log(self.base) * self.output.data
        
        return (input_grad,)
    