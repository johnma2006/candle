import numpy as np
from .module import Module
from ..tensor import Tensor


class Dropout(Module):
    
    def __init__(self,
                 p: float):
        super().__init__()
        if p is None:
            p = 0.0
        self.p = p
        
        
    def forward(self, x):
        if self.training and self.p > 0.0:
            mask = Tensor((np.random.random(size=x.shape) > self.p) / (1 - self.p))
            return x * mask
        else:
            return x
        
    def __repr__(self):
        return f'Dropout({self.p})'
