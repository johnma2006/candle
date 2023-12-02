"""Operations in a computation graph."""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import List

import candle
from .. import tensor


class Operation(ABC):
    
    def __init__(self,
                 inputs: List[Tensor]):
        """Initializes operation.
        
        Parameters
        ----------
        inputs
            List of Tensor inputs into operation.
        
        """
        # If any of the inputs are scalars, cast to Tensor
        inputs = [tensor.Tensor(x) if is_nontensor_scalar(x) else x for x in inputs]
        
        for x in inputs:
            if not isinstance(x, tensor.Tensor):
                raise ValueError(f'Input is type {type(x)}, but all inputs must be type Tensor.')
        
        self.inputs: List[Tensor] = inputs
        self.output: Tensor = None  # Result of self.forward()
        
        
    @abstractmethod
    def _forward(self):
        pass
    
    
    @abstractmethod
    def _backward(self, output_grad: np.array):
        pass
    
    
    def forward(self):
        """Computes the result of the operation, and conditionally connects result to computation graph.
        
        Returns
        -------
        output
            Tensor result of operation.
        
        """
        output = self._forward()
        
        # Conditionally connect the output node to the computation graph if any of its inputs
        # are connected to the computation graph
        is_in_computation_graph = np.any([child_node.is_in_computation_graph()
                                          for child_node in self.inputs])
        
        if is_in_computation_graph and candle.is_grad_enabled():
            output.operation = self
            self.output = output
        
        return output

         
    def backward(self, output_grad: np.array):
        """Computes the derivative of the loss node with respect to each Tensor in self.inputs.
        
        Parameters
        ----------
        output_grad
            Numpy array with shape self.output.shape.
        
        Returns
        -------
        input_grads
            List of Numpy arrays, one array of shape input.shape for each tensor `input` in self.inputs.
        
        """
        input_grads = self._backward(output_grad)
                
        # Sanity checks
        
        assert self.output is not None
        assert len(input_grads) == len(self.inputs)
        for (input_grad, inp) in zip(input_grads, self.inputs):
            assert isinstance(input_grad, np.ndarray) or np.issubdtype(input_grad, np.number)
            assert input_grad.shape == inp.shape, (
                f'input_grad.shape = {input_grad.shape} != inp.shape = {inp.shape}:'
            )
            
        return input_grads
            
        
    def free_pointers(self):
        """Remove pointers to facilitate garbage collection."""
        self.output.operation = None  # Probably only need to do this, but free the rest just in case
        self.output = None
        self.inputs = None
    
    
def is_nontensor_scalar(x):
    if isinstance(x, tensor.Tensor) or isinstance(x, np.ndarray):
        return False
    elif isinstance(x, (int, float, complex)) or np.issubdtype(x, np.number):
        return True
    else:
        return False
