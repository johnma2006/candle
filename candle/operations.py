"""Operations in a computation graph."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Type, Tuple, Union

from . import utils
from .tensor import Tensor
from .parameter import Parameter


class Operation(ABC):
    
    def __init__(self,
                 inputs: List[Tensor]):
        """Initializes operation.
        
        Parameters
        ----------
        inputs
            List of Tensor inputs into operation.
        
        """
        # If any of the inputs are scalars, cast to Tensor
        inputs = [Tensor(x) if isinstance(x, (int, float, complex, np.ndarray)) else x
                  for x in inputs]
        
        for x in inputs:
            if not isinstance(x, (Tensor, Parameter)):
                raise ValueError(f'Input is type {type(x)}, but all inputs must be type Tensor.')
        
        self.inputs = inputs
        self.output = None  # Tensor result of self.forward()
    
    
    def forward(self):
        """Computes the result of the operation.
        
        Returns
        -------
        output
            Tensor result of operation.
        
        """
        output = self._forward()
        
        output.operation = self
        self.output = output
        
        return output

         
    def backward(self, output_grad: np.array):
        """Computes the derivative of the loss node with respect to each Tensor in self.inputs.
        
        Parameters
        ----------
        output_grad
            Numpy array with shape self.output.shape.
        
        Returns
        -------
        input_grads
            List of Numpy arrays, one array of shape input.shape for each tensor `input` in self.inputs.
        
        """
        input_grads = self._backward(output_grad)
        
        assert len(input_grads) == len(self.inputs)
        for (input_grad, inp) in zip(input_grads, self.inputs):
            assert type(input_grad) is np.ndarray
            
            if input_grad.shape != inp.shape:
                raise RuntimeError(f'input_grad.shape = {input_grad.shape} != inp.shape = {inp.shape}:')
            
        return input_grads
            
        
    @abstractmethod
    def _forward(self):
        pass
    
    
    @abstractmethod
    def _backward(self,
                  output_grad: np.array):
        pass
    
    
class Addition(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data + b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Subtraction(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data - b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(-output_grad, b.shape)
        
        return (input_grad_a, input_grad_b)
        

class Multiplication(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data * b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        output_grad_a = output_grad * b.data
        output_grad_b = output_grad * a.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
        
    
class Division(Operation):
    
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(a.data / b.data)
    
    
    def _backward(self,
                  output_grad: np.array):
    
        (a, b) = self.inputs
        
        output_grad_a = output_grad / b.data
        output_grad_b = -output_grad * a.data / b.data / b.data
        
        input_grad_a = utils.sum_along_broadcasted_axes(output_grad_a, a.shape)
        input_grad_b = utils.sum_along_broadcasted_axes(output_grad_b, b.shape)
        
        return (input_grad_a, input_grad_b)
    
    
class Power(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 power: float):
        super().__init__(inputs)
        self.power = power
        
    
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(self.inputs[0].data ** self.power)
    
    
    def _backward(self,
                  output_grad: np.array):
        
        input_grad = output_grad * self.power * self.inputs[0].data ** (self.power - 1)
        return (input_grad,)
    

class TensorContraction(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 axes: int):
        super().__init__(inputs)
        self.axes = axes
        
        
    def _forward(self):
        (a, b) = self.inputs
        return Tensor(np.tensordot(a.data, b.data, axes=self.axes))
    
    
    def _backward(self,
                  output_grad: np.array):
        (a, b) = self.inputs
        
        left_dim = len(a.data.shape) - self.axes
        right_dim = len(b.data.shape) - self.axes

        input_grad_a = np.tensordot(output_grad, b.data, axes=[range(-1, -right_dim - 1, -1)] * 2)
        input_grad_b = np.tensordot(a.data, output_grad, axes=[range(left_dim)] * 2)

        return (input_grad_a, input_grad_b)
    
    
class TensorSum(Operation):
    
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
        return Tensor(self.inputs[0].data.sum(axis=self.axis, keepdims=self.keepdims))
    
    
    def _backward(self,
                  output_grad: np.array):
        if not self.keepdims:
            output_grad = np.expand_dims(output_grad, axis=self.axis)
            
        input_grad = np.broadcast_to(output_grad, shape=self.inputs[0].shape)
                         
        return (input_grad,)
    
    
class TensorSlice(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 key):
        super().__init__(inputs)
        self.key = key
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(self.inputs[0].data[self.key])
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros(self.inputs[0].shape)
        input_grad[self.key] = output_grad
        
        return (input_grad,)
    
    
class TensorTranspose(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 dim0: int,
                 dim1: int):
        super().__init__(inputs)
        self.dim0 = dim0
        self.dim1 = dim1
        
        
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(self.inputs[0].data.swapaxes(self.dim0, self.dim1))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad.swapaxes(self.dim0, self.dim1)
        
        return (input_grad,)
    

class ReLUActivation(Operation):
    
    def _forward(self):
        assert len(self.inputs) == 1
        return Tensor(np.maximum(self.inputs[0].data, 0.0))
    
    
    def _backward(self,
                  output_grad: np.array):
        input_grad = output_grad * (self.inputs[0].data > 0)
        
        return (input_grad,)
    
    
class CrossEntropyLossOperation(Operation):
    
    def _forward(self):
        (logits, target) = self.inputs
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert len(target) == len(logits)
        
        log_softmax = utils.log_softmax(logits.data)

        return Tensor(-np.mean(log_softmax[range(len(target)), target.data]))
    
    
    def _backward(self,
                  output_grad: np.array):
        (logits, target) = self.inputs
        
        softmax = utils.softmax(logits.data)
        softmax[range(len(target)), target.data] -= 1
        
        return (output_grad * softmax / len(logits), np.zeros(len(target)))  # target has no gradient
    
    
class Conv2dOperation(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0):
        
        super().__init__(inputs)
        
        if type(stride) is int:
            stride = (stride, stride)
        if type(padding) is int:
            padding = (padding, padding)
            
        assert stride[0] >= 1 and stride[1] >= 1
        assert padding[0] >= 0 and padding[1] >= 0
        
        self.stride = stride
        self.padding = padding
        
        
    def _forward(self):
        # image: np.array with shape (N, in_channels, height, width)
        # kernel: np.array with shape (in_channels, out_channels, kernel_height, kernel_width)
        (image, kernel) = self.inputs

        if len(image.shape) != 4:
            raise ValueError(f'image.shape is {image.shape}, but must be 4-dimensional (N, channels, height, width).')

        if image.shape[1] != kernel.shape[0]:
            raise ValueError(f'image.shape is {image.shape} with channel dimension {image.shape[1]}, '
                             f'but must have channel dimension {kernel.shape[0]}.')

            
        convolved_image = utils.conv2d(image.data,
                                       kernel.data,
                                       padding=self.padding,
                                       stride=self.stride)

        return Tensor(convolved_image)
    
    
    def _backward(self,
                  output_grad: np.array):
        # image: Tensor with shape (N, in_channels, height, width)
        # kernel: Tensor with shape (in_channels, out_channels, kernel_height, kernel_width)
        (image, kernel) = self.inputs

        # Undo stride
        unstrided_output_grad = np.zeros((output_grad.shape[0],
                                          output_grad.shape[1],
                                          image.shape[2] + 2 * self.padding[0] - kernel.shape[2] + 1,
                                          image.shape[3] + 2 * self.padding[1] - kernel.shape[3] + 1))

        unstrided_output_grad[:, :, ::self.stride[0], ::self.stride[1]] = output_grad

        # ------------------
        # Compute image_grad
        # ------------------

        # (out_channels, in_channels, reversed height, reversed width)
        reversed_kernel = kernel.data[:, :, ::-1, ::-1].swapaxes(0, 1)

        padded_output_grad = np.pad(
            unstrided_output_grad,
            pad_width=((0, 0),                                       # N
                       (0, 0),                                       # out_channels
                       (kernel.shape[2] - 1, kernel.shape[2] - 1),   # height
                       (kernel.shape[3] - 1, kernel.shape[3] - 1))   # width
        )

        image_grad = utils.conv2d(padded_output_grad, reversed_kernel)

        # Undo padding
        image_grad = image_grad[:, :,
                                self.padding[0]:image_grad.shape[2] - self.padding[0],
                                self.padding[1]:image_grad.shape[3] - self.padding[1]]

        # -------------------
        # Compute kernel_grad
        # -------------------

        kernel_grad = np.empty(kernel.shape)

        padded_image = np.pad(image.data,
                              pad_width=((0, 0),
                                         (0, 0),
                                         (self.padding[0], self.padding[0]),
                                         (self.padding[1], self.padding[1])))

        for i in range(kernel.shape[2]):
            for j in range(kernel.shape[3]):
                image_window = padded_image[:, :,
                                            i: padded_image.shape[2] - (kernel.shape[2] - i - 1),
                                            j: padded_image.shape[3] - (kernel.shape[3] - j - 1)]

                kernel_grad[:, :, i, j] = np.einsum('imkl,inkl', image_window, unstrided_output_grad, optimize=True)

        return (image_grad, kernel_grad)
    
    
    
class MaxPool2dOperation(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__(inputs)
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(padding) is int:
            padding = (padding, padding)
            
        self.kernel_size = kernel_size
        self.padding = padding
        
        
    def _forward(self):
        # x: shape (N, channel, height, width)
        # Pad with negative infinity
        x = np.pad(self.inputs[0].data,
                   pad_width=((0, 0),
                              (0, 0),
                              (self.padding[0], self.padding[0]),
                              (self.padding[1], self.padding[1])),
                   constant_values=-np.inf)

        # Reduce height and width until dimensions are divisible by self.kernel_size
        x = x[:,
              :,
              :x.shape[2] - x.shape[2] % self.kernel_size[0],
              :x.shape[3] - x.shape[3] % self.kernel_size[1]]

        # Reshape x such that each patch is in the last 2 dimensions
        x = (x.reshape(x.shape[0],
                       x.shape[1],
                       x.shape[2] // self.kernel_size[0],
                       self.kernel_size[0],
                       x.shape[3] // self.kernel_size[1],
                       self.kernel_size[1])
             .swapaxes(3, 4))

        # Flatten x to 2D so that we can take argmax
        output_shape = x.shape[:4]
        x = x.reshape(np.prod(output_shape), np.prod(self.kernel_size))
        
        self._argmax = np.argmax(x, axis=1)

        return Tensor(x[(np.arange(len(x)), self._argmax)].reshape(output_shape))
        

    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros((np.prod(self.output.shape), np.prod(self.kernel_size)))
        input_grad[np.arange(len(self._argmax)), self._argmax] = output_grad.flatten()

        # Reshape input_grad from 2D back to 4D
        input_grad = (input_grad
                      .reshape(self.output.shape + self.kernel_size)
                      .swapaxes(3, 4))

        input_grad = input_grad.reshape((input_grad.shape[0],
                                         input_grad.shape[1],
                                         input_grad.shape[2] * input_grad.shape[3],
                                         input_grad.shape[4] * input_grad.shape[5]))

        # Undo padding
        input_grad = input_grad[:,
                                :,
                                self.padding[0]: self.padding[0] + self.inputs[0].shape[2],
                                self.padding[1]: self.padding[1] + self.inputs[0].shape[3]]
        
        # Undo "reduce height and width until dimensions are divisible"
        input_grad = np.pad(input_grad,
                            pad_width=((0, 0),
                                       (0, 0),
                                       (0, self.inputs[0].shape[2] - input_grad.shape[2]),
                                       (0, self.inputs[0].shape[3] - input_grad.shape[3])))

        return (input_grad,)
    
    
class AvgPool2dOperation(Operation):
    
    def __init__(self,
                 inputs: List[Tensor],
                 kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__(inputs)
        
        if type(kernel_size) is int:
            kernel_size = (kernel_size, kernel_size)
        if type(padding) is int:
            padding = (padding, padding)
            
        self.kernel_size = kernel_size
        self.padding = padding
        
        
    def _forward(self):
        # x: shape (N, channel, height, width)
        # Pad with 0
        x = np.pad(self.inputs[0].data,
                   pad_width=((0, 0),
                              (0, 0),
                              (self.padding[0], self.padding[0]),
                              (self.padding[1], self.padding[1])))

        # Reduce height and width until dimensions are divisible by self.kernel_size
        x = x[:,
              :,
              :x.shape[2] - x.shape[2] % self.kernel_size[0],
              :x.shape[3] - x.shape[3] % self.kernel_size[1]]

        # Reshape x such that each patch is in the last 2 dimensions
        x = (x.reshape(x.shape[0],
                       x.shape[1],
                       x.shape[2] // self.kernel_size[0],
                       self.kernel_size[0],
                       x.shape[3] // self.kernel_size[1],
                       self.kernel_size[1])
             .swapaxes(3, 4))

        return Tensor(x.mean(axis=(4, 5)))
        

    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros((np.prod(self.output.shape), np.prod(self.kernel_size))).T
        input_grad[:] = output_grad.flatten() / np.prod(self.kernel_size)
        input_grad = input_grad.T
        
        # Reshape input_grad from 2D back to 4D
        input_grad = (input_grad
                      .reshape(self.output.shape + self.kernel_size)
                      .swapaxes(3, 4))

        input_grad = input_grad.reshape((input_grad.shape[0],
                                         input_grad.shape[1],
                                         input_grad.shape[2] * input_grad.shape[3],
                                         input_grad.shape[4] * input_grad.shape[5]))
        # Undo padding
        input_grad = input_grad[:,
                                :,
                                self.padding[0]: self.padding[0] + self.inputs[0].shape[2],
                                self.padding[1]: self.padding[1] + self.inputs[0].shape[3]]

        # Undo "Reduce height and width until dimensions are divisible"
        input_grad = np.pad(input_grad,
                            pad_width=((0, 0),
                                       (0, 0),
                                       (0, self.inputs[0].shape[2] - input_grad.shape[2]),
                                       (0, self.inputs[0].shape[3] - input_grad.shape[3])))

        return (input_grad,)
    