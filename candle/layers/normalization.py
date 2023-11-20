import numpy as np
from typing import Tuple, Union

from .module import Module
from ..tensor import Tensor
from ..parameter import Parameter

    
class BatchNorm(Module):
    
    def __init__(self,
                 axis: Union[int, Tuple[int]] = (0,),
                 momentum: float = 0.1,
                 eps: float = 1e-5):
        super().__init__()
        
        if type(axis) is int:
            axis = (axis,)
        if type(axis) is not tuple or 0 not in axis:
            raise ValueError(f'axis = {axis} must contain 0, the batch dimension.')
        
        self.axis = axis
        self.momentum = momentum
        self.eps = eps
        
        self.ema_mean = 0.0
        self.ema_var = 1.0
        
        # Defer initialization to the first forward pass once we know the shape
        self.features_shape = None
        self.W = Parameter(Tensor(np.ones(0)))
        self.b = Parameter(Tensor(np.zeros(0)))
        
        
    def forward(self, x):
        # input: shape (N, *)
        # output: shape (N, *)
        if self.features_shape is None:
            features_shape = np.array(x.shape)
            features_shape[list(self.axis)] = 1
            self.features_shape = tuple(features_shape)
            self.W.data = np.ones(self.features_shape)
            self.b.data = np.zeros(self.features_shape)

        batch_mean = x.mean(axis=self.axis, keepdims=True)
        batch_var = x.var(axis=self.axis, keepdims=True)

        if self.training:
            # Update running mean/var
            self.ema_mean = self.ema_mean * (1 - self.momentum) + batch_mean.data * self.momentum
            self.ema_var = self.ema_var * (1 - self.momentum) + batch_var.data * self.momentum

            # In training mode, use batch mean/var for normalization 
            mean = batch_mean
            var = batch_var
        else:
            # In eval mode, use running mean/var for normalization 
            mean = Tensor(self.ema_mean)
            var = Tensor(self.ema_var)

        x_normalized = (x - mean) / (var + self.eps) ** 0.5
        x_normalized = x_normalized * self.W + self.b

        return x_normalized
    
        
    def __repr__(self):
        return f'BatchNorm(axis={self.axis})'
    

class LayerNorm(Module):
    
    def __init__(self,
                 axis: Union[int, Tuple[int]],
                 eps: float = 1e-5):
        super().__init__()
        
        if type(axis) is int:
            axis = (axis,)
        if type(axis) is not tuple or 0 in axis:
            raise ValueError(f'axis = {axis} must not contain 0, the batch dimension.')
        
        self.axis = axis
        self.eps = eps
        
        # Defer initialization to the first forward pass once we know the shape
        self.features_shape = None
        self.W = Parameter(Tensor(np.ones(0)))
        self.b = Parameter(Tensor(np.zeros(0)))
        
        
    def forward(self, x):
        # input: shape (N, *)
        # output: shape (N, *)
        if self.features_shape is None:
            features_shape = np.array(x.shape)
            features_shape[[i for i in range(len(x.shape)) if i not in self.axis]] = 1
            self.features_shape = tuple(features_shape)
            self.W.data = np.ones(self.features_shape)
            self.b.data = np.zeros(self.features_shape)

        mean = x.mean(axis=self.axis, keepdims=True)
        var = x.var(axis=self.axis, keepdims=True)

        x_normalized = (x - mean) / (var + self.eps) ** 0.5
        x_normalized = x_normalized * self.W + self.b

        return x_normalized
    
        
    def __repr__(self):
        return f'LayerNorm(axis={self.axis})'
    