"""Convenience wrappers over operations for commonly used functions."""

import numpy as np
from typing import Union, Tuple

from . import operations
from .tensor import Tensor


def add(a, b):
    return operations.Addition([a, b]).forward()


def sub(a, b):
    return operations.Subtraction([a, b]).forward()


def mul(a, b):
    return operations.Multiplication([a, b]).forward()


def div(a, b):
    return operations.Division([a, b]).forward()


def pow(a, power: float):
    return operations.Power([a], power=power).forward()


def exp(a, base: float):
    return operations.Exponentiation([a], base=base).forward()


def tensordot(a, b, axes: int):
    return operations.TensorContraction([a, b], axes=axes).forward()


def bmm(a, b):
    """Multiplies two tensors of shape (A, B, C, ..., M, N) and (A, B, C, ..., N, P).
    
    Returns a tensor of shape (A, B, C, ..., M, P)."""
    return operations.BatchMatrixMultiply([a, b]).forward()


def tensorslice(a, key):
    return operations.TensorSlice([a], key=key).forward()


def reshape(a, new_shape: Tuple[int]):
    return operations.TensorReshape([a], new_shape=new_shape).forward()


def swapaxes(a, dim0: int, dim1: int):
    return operations.TensorSwapaxes([a], dim0=dim0, dim1=dim1).forward()


def transpose(a):
    return operations.TensorTranspose([a]).forward()


def sum(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):
    return operations.TensorSum([a], axis=axis, keepdims=keepdims).forward()


def mean(a,
         axis: Union[int, Tuple[int]] = None,
         keepdims: bool = False):
    if type(axis) is int:
        axis = (axis,)
    if axis is None:
        N = np.prod(a.shape)
    else:
        N = np.prod(np.array(a.shape)[list(axis)])
    
    return operations.TensorSum([a], axis=axis, keepdims=keepdims).forward() / Tensor(N)


def var(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):
    mean = a.mean(axis=axis, keepdims=True)
    var = ((a - mean) ** 2).mean(axis=axis, keepdims=keepdims)
    
    return var


def max(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):
    return operations.TensorMax([a], axis=axis, keepdims=keepdims).forward()


def min(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):
    return operations.TensorMin([a], axis=axis, keepdims=keepdims).forward()


def std(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):

    return var(a, axis, keepdims) ** 0.5


def relu(a):
    return operations.ReLUActivation([a]).forward()


def gelu(a):
    return operations.GeLUActivation([a]).forward()


def softmax(a):
    """Applies softmax along the last axis of a Tensor."""
    softmax = a.T
    softmax = softmax - softmax.max(axis=0)  # For numerical stabiility
    softmax = exp(softmax, base=np.e)
    softmax = (softmax / softmax.sum(axis=0)).T

    return softmax
    

def cross_entropy_loss(logits: np.array,
                       target: np.array):
    """Cross-entropy loss between logits and labels.
    
    Parameters
    ----------
    logits
        np.array of shape (N, num_unique_labels)
    targets
        Labels. Integer np.array of shape (N,)
        
    """
    return operations.CrossEntropyLossOperation([logits, target]).forward()


def masked_fill(a,
                mask: Tensor,
                fill_value: float):
    """Returns Tensor with masked values replaced with fill_value.

    Parameters
    ----------
    a
        Tensor to fill.
    mask
        Tensor of 1s and 0s, must be broadcastable with `a`.
        1 to fill with fill_value, 0 to leave as-is.
    fill_value
        Value to fill.

    """
    return operations.TensorMaskedFill([a], mask=mask, fill_value=fill_value).forward()
            