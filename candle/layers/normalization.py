import numpy as np
from typing import Tuple, Union

from .module import Module
from ..tensor import Tensor, Parameter


class BatchNorm(Module):
    
    def __init__(self,
                 normalized_shape: Union[int, Tuple[int]], 
                 axis: Union[int, Tuple[int]] = (0, 2, 3),
                 momentum: float = 0.1,
                 eps: float = 1e-5):
        """Batch normalization.
        
        Parameters
        ----------
        normalized_shape
            The input shape for all axes not in axis.
            For example, if input.shape = (2, 3, 5, 7) and axis=(0, 2), then normalized_shape = (3, 7). 
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
        if 0 not in axis:
            raise ValueError(f'axis = {axis} must contain 0, the batch dimension.')
            
        if type(normalized_shape) is int:
            normalized_shape = (normalized_shape,)
            
        self.ema_mean = 0.0
        self.ema_var = 1.0
        
        self.axis = axis
        self.momentum = momentum
        self.eps = eps
        
        self.normalized_shape = tuple(normalized_shape)

        input_dim = len(self.axis) + len(self.normalized_shape)
        mask = np.ones(input_dim, dtype=bool)
        mask[list(self.axis)] = False
        broadcastable_shape = np.ones(input_dim).astype(int)
        broadcastable_shape[mask] = self.normalized_shape
        self.W = Parameter(np.ones(broadcastable_shape))
        self.b = Parameter(np.zeros(broadcastable_shape))
        
        
    def forward(self, x):
        # input: shape (N, *)
        # output: shape (N, *)
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
        return f'BatchNorm(normalized_shape={self.normalized_shape}, axis={self.axis})'


class LayerNorm(Module):
    
    def __init__(self,
                 normalized_shape: Union[int, Tuple[int]], 
                 axis: Union[int, Tuple[int]] = None,
                 eps: float = 1e-5):
        """Layer normalization.
        
        Parameters
        ----------
        normalized_shape
            The input shape for all axes in axis.
            For example, if input.shape = (2, 3, 5, 7) and axis=(-1, -2), then normalized_shape = (5, 7). 
        axis
            Axes to compute mean/std over. Must not include the 0, batch axis.
            If None, then normalizes over the last len(normalized_shape) dimensions.
        eps
            Value added to the denominator for numerical stability.
            
        """
        super().__init__()
        
        if type(normalized_shape) is int:
            normalized_shape = (normalized_shape,)
        if axis is None:
            axis = tuple([-i for i in range(1, 1 + len(normalized_shape))])
        elif type(axis) is int:
            axis = (axis,)
        if 0 in axis:
            raise ValueError(f'axis = {axis} must not contain 0, the batch dimension.')
        
        self.axis = axis
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)

        input_dim = len(self.axis) + len(self.normalized_shape)
        mask = np.zeros(input_dim, dtype=bool)
        mask[list(self.axis)] = True
        broadcastable_shape = np.ones(input_dim).astype(int)
        broadcastable_shape[mask] = self.normalized_shape
        self.W = Parameter(np.ones(broadcastable_shape))
        self.b = Parameter(np.zeros(broadcastable_shape))
        
        
    def forward(self, x):
        # input: shape (N, *)
        # output: shape (N, *)
        mean = x.mean(axis=self.axis, keepdims=True)
        var = x.var(axis=self.axis, keepdims=True)

        x_normalized = (x - mean) / (var + self.eps) ** 0.5
        x_normalized = x_normalized * self.W + self.b

        return x_normalized
    
        
    def __repr__(self):
        return f'LayerNorm(normalized_shape={self.normalized_shape}, axis={self.axis})'


class RMSNorm(Module):
    
    def __init__(self,
                 normalized_shape: Union[int, Tuple[int]], 
                 axis: Union[int, Tuple[int]] = None,
                 eps: float = 1e-5):
        """RMS normalization.
        
        Parameters
        ----------
        normalized_shape
            The input shape for all axes in axis.
            For example, if input.shape = (2, 3, 5, 7) and axis=(1, 2), then normalized_shape = (3, 5). 
        axis
            Axes to compute RMS over. Must not include the 0, batch axis.
            If None, then normalizes over the last len(normalized_shape) dimensions.
        eps
            Value added to the denominator for numerical stability.
            
        """
        super().__init__()
        
        if type(normalized_shape) is int:
            normalized_shape = (normalized_shape,)
        if axis is None:
            axis = tuple([-i for i in range(1, 1 + len(normalized_shape))])
        elif type(axis) is int:
            axis = (axis,)
        if 0 in axis:
            raise ValueError(f'axis = {axis} must not contain 0, the batch dimension.')
        
        self.axis = axis
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)

        input_dim = len(self.axis) + len(self.normalized_shape)
        mask = np.zeros(input_dim, dtype=bool)
        mask[list(self.axis)] = True
        broadcastable_shape = np.ones(input_dim).astype(int)
        broadcastable_shape[mask] = self.normalized_shape
        self.W = Parameter(np.ones(broadcastable_shape))
        
        
    def forward(self, x):
        # input: shape (N, *)
        # output: shape (N, *)
        rms = (x ** 2).mean(axis=self.axis, keepdims=True)

        x_normalized = x / (rms + self.eps) ** 0.5
        x_normalized = x_normalized * self.W

        return x_normalized
    
        
    def __repr__(self):
        return f'RMSNorm(normalized_shape={self.normalized_shape}, axis={self.axis})'
    