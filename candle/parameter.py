import numpy as np
import pandas as pd
from typing import List, Tuple
from abc import ABC, abstractmethod

from .tensor import Tensor


class Parameter:
    """Wrapper around Tensors that indicates that it should be updated during backprop."""
    
    def __init__(self,
                 weight: Tensor):
        self.weight = weight
        self.weight.requires_grad = True
        

    def __getattr__(self, attr):
        return getattr(self.weight, attr)
    
    
    def __setattr__(self, name, value):
        if name == 'weight':
            object.__setattr__(self, name, value)  # Call original __setattr__
        else:
            setattr(self.weight, name, value)
            

    def __repr__(self):
        return str(self.weight).replace('Tensor(', f'Parameter(')
    
    # -----------------------------
    # Redefine operation overloads
    # (there is no way around this)
    # -----------------------------
    
    def __getitem__(self, key):
        return self.weight.__getitem__(key)
    
    
    def __add__(self, other):
        return self.weight.__add__(other)
    
    
    def __radd__(self, other):
        return self.weight.__radd__(other)
    
    
    def __sub__(self, other):
        return self.weight.__sub__(other)
    
    
    def __rsub__(self, other):
        return self.weight.__rsub__(other)
    

    def __mul__(self, other):
        return self.weight.__mul__(other)
    
    
    def __rmul__(self, other):
        return self.weight.__rmul__(other)
    
    
    def __truediv__(self, other):
        return self.weight.__truediv__(other)
    
    
    def __rtruediv__(self, other):
        return self.weight.__rtruediv__(other)
    
    
    def __neg__(self):
        return self.weight.__neg__()
    
    
    def __pos__(self):
        return self.weight.__pos__()
    
    
    def __pow__(self, power):
        return self.weight.__pow__(power)
    
    
    def __matmul__(self, other):
        return self.weight.__matmul__(other)
    