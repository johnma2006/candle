"""Tensor and autograd functionality."""

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
            
        self.data = data.astype(dtype)
        
        self.grad = 0.0
        self.operation = None  # Operation edge whose result is this Tensor node. None if is leaf
        self.requires_grad = False
        
        self._batch_grad = 0.0  # Grads for each batch during loss.backward() accumulate here.
        self._outdegree = 0     # Outdegree in the computation graph. Uninitialized until .backward()
        self._requires_grad_computation = None  # If we need to compute grad during backprop.
                                                # True iff any child node is True.
                                                # Uninitialized until _initialize_requires_grad_computation()
    
    # ---------------
    # Backpropagation
    # ---------------
    
    def zero_grad(self):
        """Resets grad to 0."""
        self.grad = 0.0
        
    
    def backward(self):
        """Computes grad w.r.t. this tensor for all upstream nodes in the computation graph."""
        if self.shape != ():
            raise Exception('.backward() can only be called from a scalar Tensor.')
        
        self._reset_graph()
        self._initialize_outdegree()
        self._initialize_requires_grad_computation()
        
        self._batch_grad = np.array(1.0, dtype=Tensor.DEFAULT_DTYPE)
        self._backward()

    
    def _backward(self):
        """Accumulates self.grad, the derivative w.r.t. the loss tensor, for this and all upstream nodes."""
        if self.operation is None:  # Is leaf
            return
        
        nodes_to_backprop = []
        input_grads = self.operation.backward(self._batch_grad)
        self._batch_grad = 0.0  # Free memory
        
        for (node, input_grad) in zip(self.operation.inputs, input_grads):
            if node._requires_grad_computation:
                node._batch_grad += input_grad
                if node.requires_grad:
                    node.grad += input_grad
                
                node._outdegree -= 1
                assert node._outdegree >= 0

                if node._outdegree == 0:
                    nodes_to_backprop.append(node)
                    
        self.operation.free_memory()  # Remove pointers to facilitate garbage collection
        
        for node in set(nodes_to_backprop):
            node._backward()
        
        
    def _reset_graph(self):
        """Resets the computation graph in preparation for backprop.
        
        Resets self._outdegree, self._requires_grad_computation, and self.batch_grad.
        
        """
        children = [self]
        seen = set()

        while children:
            node = children.pop(0)
            node._batch_grad = 0.0
            node._outdegree = 0
            node._requires_grad_computation = None

            if node.operation is not None:
                for child_node in set(node.operation.inputs):
                    if id(child_node) not in seen:
                        children.append(child_node)

            seen.add(id(node))
        
                
    def _initialize_outdegree(self):
        """Initializes self._outdegree for this node and upstream nodes."""
        children = [self]
        seen = set()

        while children:
            node = children.pop(0)

            if node.operation is not None:
                for child_node in node.operation.inputs:
                    child_node._outdegree += 1

                for child_node in set(node.operation.inputs):
                    if id(child_node) not in seen:
                        children.append(child_node)

            seen.add(id(node))
            
            
    def _initialize_requires_grad_computation(self):
        """Initializes self._requires_grad_computation for this node and upstream nodes."""
        if self._requires_grad_computation is not None:
            return self._requires_grad_computation
        
        if self.operation is None:  # Is leaf
            self._requires_grad_computation = self.requires_grad
        else:
            self._requires_grad_computation = False
            for child_node in set(self.operation.inputs):
                self._requires_grad_computation |= child_node._initialize_requires_grad_computation()

        return self._requires_grad_computation

    # ---------------------
    # General functionality
    # ---------------------
    
    @property
    def shape(self):
        """Tuple of array dimensions."""
        return self.data.shape
    
    
    @property
    def dtype(self):
        """Returns dtype of Tensor."""
        return self.data.dtype
    
    
    def astype(self, dtype):
        """Casts Tensor to dtype.
        
        Parameters
        ----------
        dtype
            Numpy dtype.
            
        """
        return Tensor(self.data.copy(), dtype=dtype)
    
    
    def clone(self):
        """Returns clone of tensor with same relationship to the computation graph."""
        cloned = Tensor(self.data.copy(), dtype=self.data.dtype)
        cloned.operation = self.operation
        cloned.requires_grad = self.requires_grad
    
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
    
    
    def flatten(self):
        return functions.reshape(self, new_shape=(-1,))
    
    # -------------------
    # Operation overloads
    # -------------------
    
    def __getitem__(self, key):
        return functions.tensorslice(self, key)
    
    
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
            Dtype of tensor. If None, autocasts to float32.
            
        """
        super().__init__(data, dtype)
        self.requires_grad = True
        
        
    def __repr__(self):
        if len(self.shape) == 0:
            return self.data.__repr__().replace('array', 'Parameter')
        else:
            return f'Parameter({self.shape})-shape {str(self.data.dtype)} array)'
  
        