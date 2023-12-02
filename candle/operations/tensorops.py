from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union

from .operation import Operation
from .. import tensor


class TensorContraction(Operation):
    """f(inputs) = tensordot(inputs[0], inputs[1], axes)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 axes: int):
        super().__init__(inputs)
        self.axes = axes
        
        
    def _forward(self):
        (a, b) = self.inputs
        return tensor.Tensor(np.tensordot(a.data, b.data, axes=self.axes))
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        left_dim = len(a.data.shape) - self.axes
        right_dim = len(b.data.shape) - self.axes

        input_grad_a = np.tensordot(output_grad, b.data, axes=[range(-1, -right_dim - 1, -1)] * 2)
        input_grad_b = np.tensordot(a.data, output_grad, axes=[range(left_dim)] * 2)

        return (input_grad_a, input_grad_b)
    
    
class TensorSum(Operation):
    """f(inputs) = inputs[0].sum(axis, keepdims)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 axis: Union[int, Tuple[int, int]] = None,
                 keepdims: bool = False):
        super().__init__(inputs)
        if type(axis) is int:
            axis = (axis,)
        if axis is None:
            axis = tuple(range(len(self.inputs[0].shape)))
        
        self.axis = axis
        self.keepdims = keepdims
    
    
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.sum(axis=self.axis, keepdims=self.keepdims))
    
    
    def _backward(self,
                  output_grad: np.array):
        if not self.keepdims:
            output_grad = np.expand_dims(output_grad, axis=self.axis)
            
        input_grad = np.broadcast_to(output_grad, shape=self.inputs[0].shape)
                         
        return (input_grad,)
    
    
class TensorMax(Operation):
    """f(inputs) = inputs[0].max(axis, keepdims)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 axis: Union[int, Tuple[int, int]] = None,
                 keepdims: bool = False):
        super().__init__(inputs)
        if type(axis) is int:
            axis = (axis,)
        if axis is None:
            axis = tuple(range(len(self.inputs[0].shape)))
        
        self.axis = axis
        self.keepdims = keepdims
    
    
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.max(axis=self.axis, keepdims=self.keepdims))
    
    
    def _backward(self,
                  output_grad: np.array):
        output = self.output.data

        if not self.keepdims:
            output_grad = np.expand_dims(output_grad, axis=self.axis)
            output = np.expand_dims(output, axis=self.axis)

        mask = self.inputs[0].data == np.broadcast_to(output, self.inputs[0].shape)

        input_grad = mask * np.broadcast_to(output_grad, shape=self.inputs[0].shape)
                         
        return (input_grad,)
    
    
class TensorMin(Operation):
    """f(inputs) = inputs[0].min(axis, keepdims)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 axis: Union[int, Tuple[int, int]] = None,
                 keepdims: bool = False):
        super().__init__(inputs)
        if type(axis) is int:
            axis = (axis,)
        if axis is None:
            axis = tuple(range(len(self.inputs[0].shape)))
        
        self.axis = axis
        self.keepdims = keepdims
    
    
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.min(axis=self.axis, keepdims=self.keepdims))
    
    
    def _backward(self,
                  output_grad: np.array):
        output = self.output.data

        if not self.keepdims:
            output_grad = np.expand_dims(output_grad, axis=self.axis)
            output = np.expand_dims(output, axis=self.axis)

        mask = self.inputs[0].data == np.broadcast_to(output, self.inputs[0].shape)

        input_grad = mask * np.broadcast_to(output_grad, shape=self.inputs[0].shape)
                         
        return (input_grad,)
    
    
class TensorSlice(Operation):
    """f(inputs) = inputs[0][key]"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 key):
        super().__init__(inputs)
        if type(key) is not tuple:
            key = (key,)
            
        # Numpy slicing is quite involved and it's hard to cover every edge case
        # For now, we can guarantee backprop supports slicing with ints, slices, and a single leading 2D list,
        # e.g. x[[[0, 1, 2], [5, 2, 3]], 2, :, 2:5:-1]. This covers most practical cases.
        for key_i in key[1:]:
            assert type(key_i) in [int, slice]
        
        if type(key[0]) is list:
            self.list_ndim = np.array(key[0]).ndim  # Check that leading list is at most 2D
        else:
            self.list_ndim = 0
        assert self.list_ndim <= 2
        
        self.key = key
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data[self.key])
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros(self.inputs[0].shape, dtype=tensor.Tensor.DEFAULT_DTYPE)

        if self.list_ndim <= 1:
            input_grad[self.key] = output_grad

        else:  # list_ndim == 2
            for (i, subarray) in enumerate(self.key[0]):
                input_grad[(self.key[0][i],) + self.key[1:]] += output_grad[i]

        return (input_grad,)
    
    
class TensorReshape(Operation):
    """f(inputs) = inputs[0].reshape(new_shape)"""

    def __init__(self,
                 inputs: List[Tensor],
                 new_shape: Tuple[int]):
        super().__init__(inputs)
        self.new_shape = new_shape
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.reshape(self.new_shape))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad.reshape(self.inputs[0].shape)
        
        return (input_grad,)
    
    
class TensorSwapaxes(Operation):
    """f(inputs) = inputs[0].swapaxes(dim0, dim1)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 dim0: int,
                 dim1: int):
        super().__init__(inputs)
        self.dim0 = dim0
        self.dim1 = dim1
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.swapaxes(self.dim0, self.dim1))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad.swapaxes(self.dim0, self.dim1)
        
        return (input_grad,)
    
    
class BatchMatrixMultiply(Operation):
    """Multiplies two tensors of shape (A, B, C, ..., M, N) and (A, B, C, ..., N, P).

    Returns a tensor of shape (A, B, C, ..., M, P)."""
    
    def __init__(self,
                 inputs: List[Tensor]):
        super().__init__(inputs)
        
        
    def _forward(self):
        assert len(self.inputs) == 2
        (a, b) = self.inputs
        assert a.shape[:-2] == b.shape[:-2]  # Assert first N-2 dimensions match

        return tensor.Tensor(np.einsum('...ij, ...jk -> ...ik', a.data, b.data))
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        input_grad_a = np.einsum('...ij, ...kj -> ...ik', output_grad, b.data)
        input_grad_b = np.einsum('...ij, ...ik -> ...kj', output_grad, a.data)
        
        return (input_grad_a, input_grad_b)
    
    
class TensorTranspose(Operation):
    """f(inputs) = inputs[0].T"""
    
    def __init__(self,
                 inputs: List[Tensor]):
        super().__init__(inputs)
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.T)
    
    
    def _backward(self,
                  output_grad: np.array):
        return (output_grad.T,)
    
    
class TensorMaskedFill(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 mask: Tensor,
                 fill_value: float):
        """Returns Tensor with masked values replaced with fill_value.
        
        Parameters
        ----------
        inputs
            Single Tensor.
        mask
            Tensor of 1s and 0s, must be broadcastable with inputs[0].
            1 to fill with fill_value, 0 to leave as-is.
        fill_value
            Value to fill.
            
        """
        super().__init__(inputs)
        if not np.alltrue(np.isclose(mask.data, 0) | np.isclose(mask.data, 1)):
            raise ValueError('mask must be a Tensor with only 0s and 1s.')
        self.broadcasted_mask = np.broadcast_to(mask.data, self.inputs[0].shape)
        self.fill_value = fill_value

        
    def _forward(self):
        assert len(self.inputs) == 1
        
        return tensor.Tensor((1 - self.broadcasted_mask) * self.inputs[0].data + self.broadcasted_mask * self.fill_value)
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad * (1 - self.broadcasted_mask)
        
        return (input_grad,)
    
    
class TensorConcatenation(Operation):
    """f(inputs) = np.concatenate(*inputs[0], axis=axis)"""
    
    def __init__(self,
                 inputs: List[Tensor],
                 axis: int = 0):
        super().__init__(inputs)
        self.axis = axis
    
    
    def _forward(self):
        return tensor.Tensor(np.concatenate([tensor.data for tensor in self.inputs], axis=self.axis))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_lengths_along_axis = [i.shape[self.axis] for i in self.inputs]
        indices_or_sections = np.cumsum(input_lengths_along_axis)[:-1]

        input_grads = tuple(np.split(output_grad, indices_or_sections, axis=self.axis))

        return input_grads
    
    
class TensorClone(Operation):
    """f(inputs) = inputs[0].copy()"""
    
    def __init__(self,
                 inputs: List[Tensor]):
        super().__init__(inputs)
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return tensor.Tensor(self.inputs[0].data.copy())
    
    
    def _backward(self,
                  output_grad: np.array):
        return (output_grad,)
    