import numpy as np

from .module import Module
from .. import weightinit as init
from ..tensor import Tensor, Parameter, rand
    

class Linear(Module):
    
    def __init__(self,
                 input_nodes: int,
                 output_nodes: int,
                 bias: bool = True):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = bias
        
        self.W = Parameter(Tensor(init.kaiming((input_nodes, output_nodes))))
        if bias:
            k = 1.0 / np.sqrt(input_nodes)
            self.b = Parameter(rand(output_nodes, a=-k, b=k))
        
    def forward(self, x):
        if self.bias:
            return x @ self.W + self.b
        else:
            return x @ self.W
        
        
    def __repr__(self):
        return f'Linear({self.input_nodes}, {self.output_nodes})'
    