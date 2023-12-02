import numpy as np
from typing import Tuple, Union

from .module import Module
from ..tensor import Tensor, Parameter

    
class BatchNorm(Module):
    
    def __init__(self,
                 axis: Union[int, Tuple[int]] = (0,),
                 momentum: float = 0.1,
                 eps: float = 1e-5):
        """Batch normalization.
        
        Parameters
        ----------
        axis
            Axis to compute mean/std over. Must include the 0, batch axis.
            For example, axis=(0,) for BatchNorm1d. axis=(0, 2, 3) for BatchNorm2d.
        momentum
            How fast to decay the running_mean and running_var estimations.
        eps
            Value added to the denominator for numerical stability.
            
        """
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
            self.W.data = np.ones(self.features_shape, dtype=self.W.dtype)
            self.b.data = np.zeros(self.features_shape, dtype=self.b.dtype)

        if self.training:
            batch_mean = x.mean(axis=self.axis, keepdims=True)
            batch_var = x.var(axis=self.axis, keepdims=True)  # The computes biased var
            
            # Update running mean/var, making sure to update ema_var with unbiased var
            N = np.prod(np.array(x.shape)[list(self.axis)])
            bessel_correction = N / (N - 1)
            self.ema_mean = self.ema_mean * (1 - self.momentum) + batch_mean.data * self.momentum
            self.ema_var = self.ema_var * (1 - self.momentum) + batch_var.data * self.momentum * bessel_correction
            
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
        """Layer normalization.
        
        Parameters
        ----------
        axis
            Axes to compute mean/std over. Must not include the 0, batch axis.
        eps
            Value added to the denominator for numerical stability.
            
        """
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
            self.W.data = np.ones(self.features_shape, dtype=self.W.dtype)
            self.b.data = np.zeros(self.features_shape, dtype=self.b.dtype)

        mean = x.mean(axis=self.axis, keepdims=True)
        var = x.var(axis=self.axis, keepdims=True)

        x_normalized = (x - mean) / (var + self.eps) ** 0.5
        x_normalized = x_normalized * self.W + self.b

        return x_normalized
    
        
    def __repr__(self):
        return f'LayerNorm(axis={self.axis})'
    