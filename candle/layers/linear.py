import numpy as np

from .module import Module
from .. import utils
from ..tensor import Tensor, Parameter
    

class Linear(Module):
    
    def __init__(self,
                 input_nodes: int,
                 output_nodes: int,
                 bias: bool = True):
        super().__init__()
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.bias = bias
        
        k = 1.0 / np.sqrt(input_nodes)
        self.W = Parameter(Tensor(utils.kaiming_init(input_nodes, output_nodes)))
        if bias:
            self.b = Parameter(Tensor(np.random.uniform(low=-k, high=k, size=output_nodes)))
        
        
    def forward(self, x):
        if self.bias:
            return x @ self.W + self.b
        else:
            return x @ self.W
        
        
    def __repr__(self):
        return f'Linear({self.input_nodes}, {self.output_nodes})'
    