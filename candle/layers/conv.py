import numpy as np
from typing import Tuple, Union

from .module import Module
from .. import operations
from ..tensor import Tensor, Parameter
    
    
class Conv2d(Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        k = 1 / np.sqrt(in_channels * np.prod(kernel_size))
        self.kernel = Parameter(Tensor(np.random.uniform(-k, k, size=(in_channels, out_channels,
                                                                      kernel_size[0], kernel_size[1]))))
        self.bias = Parameter(Tensor(np.random.uniform(-k, k, size=(1, out_channels, 1, 1))))
        
        
    def forward(self, x):
        # input: shape (N, in_channels, height, width)
        # output: shape (N, out_channels, height, width)
        operation = operations.Conv2dOperation(inputs=[x, self.kernel],
                                               stride=self.stride,
                                               padding=self.padding)
        
        return operation.forward() + self.bias
        

    def __repr__(self):
        stride_str = f', stride={self.stride}' if self.stride != 1 else ''
        padding_str = f', padding={self.padding}' if self.padding != 0 else ''
        
        return (f'Conv2d(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}'
                f'{stride_str}{padding_str})')
    
    
class MaxPool2d(Module):
    
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        
        
    def forward(self, x):
        # input: shape (N, in_channels, height, width)
        # output: shape (N, in_channels, height, width)
        operation = operations.MaxPool2dOperation(inputs=[x],
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding)
        
        return operation.forward()
    
        
    def __repr__(self):
        padding_str = f', padding={self.padding}' if self.padding != 0 else ''
        return (f'MaxPool2d(kernel_size={self.kernel_size}{padding_str})')
    
    
    
class AvgPool2d(Module):
    
    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.padding = padding
        
        
    def forward(self, x):
        # input: shape (N, in_channels, height, width)
        # output: shape (N, in_channels, height, width)
        operation = operations.AvgPool2dOperation(inputs=[x],
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding)
        
        return operation.forward()
    
        
    def __repr__(self):
        padding_str = f', padding={self.padding}' if self.padding != 0 else ''
        return (f'AvgPool2d(kernel_size={self.kernel_size}{padding_str})')
    