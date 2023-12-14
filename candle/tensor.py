"""Tensor and autograd functionality."""

from __future__ import annotations
import numpy as np
from typing import Union, Tuple

from . import functions


class Tensor:
    """Tensor node in the computation graph."""

    DEFAULT_DTYPE = np.float32

    
    def __init__(self,
                 data: np.array,
                 dtype: type = None):
        """Initialize Tensor.

        Parameters
        ----------
        data
            Numpy array.
        dtype
            dtype of tensor. If None, autocasts to Tensor.DEFAULT_DTYPE (float32).
            
        """
        if dtype is None:
            dtype = self.DEFAULT_DTYPE

        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
            
        self.data: np.array = data.astype(dtype)
        self.grad: np.array = None
        self.operation: Operation = None  # Operation edge whose result is this Tensor node
        self.requires_grad = False
        self._retain_grad = False
        
    # ---------------
    # Backpropagation
    # ---------------        
    
    def is_in_computation_graph(self):
        """Returns True if this node is in the computation graph."""
        # If self.operation is not None, we know that one of its upstream children
        # has requires_grad == True
        return self.operation is not None or self.requires_grad
        
    
    def backward(self):
        """Populates node.grad with respect to this Tensor for all nodes in the computation graph."""
        if self.shape != ():
            raise RuntimeError('.backward() can only be called from a scalar Tensor.')
            
        if self.operation is None:
            raise RuntimeError('Empty computation graph. Either no Tensors have requires_grad == True, '
                               'or backward() was called a second time, but the intermediate '
                               'activations have already been freed.')
        
        def topological_sort(node):
            seen.add(id(node))
            if node.operation is not None:
                for child_node in node.operation.inputs:
                    if child_node.is_in_computation_graph() and id(child_node) not in seen:
                        topological_sort(child_node)
            topologically_sorted_graph.append(node)

        seen = set()
        topologically_sorted_graph = []
        topological_sort(self)
        
        self.grad = np.array(1.0, dtype=Tensor.DEFAULT_DTYPE)

        # Backprop in reverse order through the topologically sorted list of nodes
        # Since the list is topologically sorted, we know that upon reaching node N,
        # all parents of N have already been backpropped through
        for node in topologically_sorted_graph[::-1]:
            if node.operation is None:  # If leaf node, do nothing
                assert node.requires_grad
                continue

            assert not node.requires_grad
            input_grads = node.operation.backward(node.grad)
            if not node._retain_grad:
                node.grad = None  # Free memory

            for (child_node, input_grad) in zip(node.operation.inputs, input_grads):
                if child_node.is_in_computation_graph():
                    if child_node.grad is None:
                        child_node.grad = 0.0
                        
                    child_node.grad += input_grad

            node.operation.free_pointers()  # Delete pointers to facilitate garbage collection

    # ---------------------
    # General functionality
    # ---------------------

    def retain_grad(self):
        """Allows the tensor to have its grad populated during backward()."""
        self._retain_grad = True

    
    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self.data.shape
    
    
    @property
    def dtype(self):
        """Returns dtype of Tensor."""
        return self.data.dtype
    
    
    def astype(self, dtype):
        """Returns cloned Tensor casted to dtype.
        
        Parameters
        ----------
        dtype
            Numpy dtype.
            
        """
        cloned = self.clone()
        cloned.data = cloned.data.astype(dtype)
        
        return cloned
        
        
    def __repr__(self):
        if len(self.shape) == 0:
            return self.data.__repr__().replace('array', 'Tensor')
        else:
            return f'Tensor({self.shape})-shape {str(self.data.dtype)} array)'
    
    
    def __len__(self):
        return self.shape[0]
    
    # ----------
    # Operations
    # ----------
    
    def sum(self,
            axis: Union[int, Tuple[int]] = None,
            keepdims: bool = False):
        return functions.sum(self, axis=axis, keepdims=keepdims)
    
    
    def mean(self,
             axis: Union[int, Tuple[int]] = None,
             keepdims: bool = False):
        return functions.mean(self, axis=axis, keepdims=keepdims)
    
    
    def var(self,
            axis: Union[int, Tuple[int]] = None,
            keepdims: bool = False):
        return functions.var(self, axis=axis, keepdims=keepdims)
    
    
    def std(self,
            axis: Union[int, Tuple[int]] = None,
            keepdims: bool = False):
        return functions.std(self, axis=axis, keepdims=keepdims)
    
    
    def max(self,
            axis: Union[int, Tuple[int]] = None,
            keepdims: bool = False):
        return functions.max(self, axis=axis, keepdims=keepdims)
    
    
    def min(self,
            axis: Union[int, Tuple[int]] = None,
            keepdims: bool = False):
        return functions.min(self, axis=axis, keepdims=keepdims)
    
    
    def transpose(self,
                  dim0: int,
                  dim1: int):
        return functions.swapaxes(self, dim0=dim0, dim1=dim1)
    
    
    @property
    def T(self):
        return functions.transpose(self)
    
    
    def reshape(self,
                new_shape: Tuple[int]):
        return functions.reshape(self, new_shape=new_shape)

    
    def unsqueeze(self,
                  axis: int):
        return functions.unsqueeze(self, axis=axis)
    
    
    def flatten(self):
        return functions.reshape(self, new_shape=(-1,))
    
    
    def repeat_interleave(self,
                          repeats: int,
                          axis: int):
        return functions.repeat_interleave(self, repeats, axis)

    
    def clone(self):
        return functions.clone(self)
    
    # ---------
    # Overloads
    # ---------
    
    def __getitem__(self, key):
        return functions.tensorslice(self, key)


    def __setitem__(self, key, value):
        # Modify computation graph from other->self to other->old_self->new_self
        # TODO: implement _version, disallow grads + inplace modification
        #       also, this implementation is a bit hacky feeling.
        new_self = functions.tensorset(self, key, value)

        old_self = Tensor(self.data)
        old_self._retain_grad = self._retain_grad
        if self.operation is not None:
            old_self.operation = self.operation
            old_self.operation.output = old_self
        
        self.data = new_self.data
        self.operation = new_self.operation
        self.operation.output = self
        self.operation.inputs[0] = old_self


    def __add__(self, other):
        return functions.add(self, other)
    
    
    def __radd__(self, other):
        return functions.add(self, other)
    
    
    def __sub__(self, other):
        return functions.sub(self, other)
    
    
    def __rsub__(self, other):
        return functions.sub(other, self)
    

    def __mul__(self, other):
        return functions.mul(self, other)
    
    
    def __rmul__(self, other):
        return functions.mul(self, other)
    
    
    def __truediv__(self, other):
        return functions.div(self, other)
    
    
    def __rtruediv__(self, other):
        return functions.div(other, self)
    
    
    def __neg__(self):
        return functions.sub(0.0, self)
    
    
    def __pos__(self):
        return self
    
    
    def __pow__(self, power):
        return functions.pow(self, power)
    
    
    def __rpow__(self, base):
        return functions.exp(self, base)
    
    
    def __matmul__(self, other):
        return functions.tensordot(self, other, axes=1)


    def __eq__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.data == other.data).astype(bool)
            
    
class Parameter(Tensor):
    """Wrapper around Tensors that indicates that it should be updated during backprop."""
    
    def __init__(self,
                 data: np.array,
                 dtype: type = None):
        """Initialize Parameter
        
        Parameters
        ----------
        data
            Numpy array.
        dtype
            dtype of tensor. If None, autocasts to float32.
            
        """
        super().__init__(data, dtype)
        self.requires_grad = True
        
        
    def __repr__(self):
        if len(self.shape) == 0:
            return self.data.__repr__().replace('array', 'Parameter')
        else:
            return f'Parameter({self.shape})-shape {str(self.data.dtype)} array)'


def rand(*size: Tuple[int]):
    if type(size[0]) in [tuple, list]:
        assert len(size) == 1
        size = size[0]
    return Tensor(np.random.random(size=size))


def randn(*size: Tuple[int]):
    if type(size[0]) in [tuple, list]:
        assert len(size) == 1
        size = size[0]
    return Tensor(np.random.normal(size=size))


def zeros_like(tensor: Tensor):
    return Tensor(np.zeros(tensor.shape))


def ones_like(tensor: Tensor):
    return Tensor(np.ones(tensor.shape))


def empty_like(tensor: Tensor):
    return Tensor(np.empty(tensor.shape))
    