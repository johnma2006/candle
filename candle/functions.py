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


def tensordot(a, b, axes: int):
    return operations.TensorContraction([a, b], axes=axes).forward()


def bmm(a, b):
    """Multiplies two tensors of shape (A, B, C, ..., M, N) and (A, B, C, ..., N, P).
    
    Returns a tensor of shape (A, B, C, ..., M, P)."""
    return operations.BatchMatrixMultiply([a, b]).forward()


def tensorslice(a, key):
    return operations.TensorSlice([a], key=key).forward()


def transpose(a, dim0: int, dim1: int):
    return operations.TensorTranspose([a], dim0=dim0, dim1=dim1).forward()


def sum(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):
    return operations.TensorSum([a], axis=axis, keepdims=keepdims).forward()


def mean(a,
         axis: Union[int, Tuple[int]] = None,
         keepdims: bool = False):
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


def std(a,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False):

    return var(a, axis, keepdims) ** 0.5
    

def relu(a):
    return operations.ReLUActivation([a]).forward()


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
