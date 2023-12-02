from __future__ import annotations
import numpy as np
from typing import List, Tuple, Union

from .operation import Operation
from .. import tensor, utils


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

        return tensor.Tensor(convolved_image)
    
    
    def _backward(self,
                  output_grad: np.array):
        # image: Tensor with shape (N, in_channels, height, width)
        # kernel: Tensor with shape (in_channels, out_channels, kernel_height, kernel_width)
        (image, kernel) = self.inputs

        # Undo stride
        unstrided_output_grad = np.zeros((output_grad.shape[0],
                                          output_grad.shape[1],
                                          image.shape[2] + 2 * self.padding[0] - kernel.shape[2] + 1,
                                          image.shape[3] + 2 * self.padding[1] - kernel.shape[3] + 1),
                                         dtype=tensor.Tensor.DEFAULT_DTYPE)

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

        image_grad = utils.conv2d(padded_output_grad, reversed_kernel, dtype=tensor.Tensor.DEFAULT_DTYPE)

        # Undo padding
        image_grad = image_grad[:, :,
                                self.padding[0]:image_grad.shape[2] - self.padding[0],
                                self.padding[1]:image_grad.shape[3] - self.padding[1]]

        # -------------------
        # Compute kernel_grad
        # -------------------

        kernel_grad = np.empty(kernel.shape, dtype=tensor.Tensor.DEFAULT_DTYPE)

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

        return tensor.Tensor(x[(np.arange(len(x)), self._argmax)].reshape(output_shape))
        

    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros((np.prod(self.output.shape), np.prod(self.kernel_size)),
                              dtype=tensor.Tensor.DEFAULT_DTYPE)
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

        return tensor.Tensor(x.mean(axis=(4, 5)))
        

    def _backward(self,
                  output_grad: np.array):
        input_grad = np.zeros((np.prod(self.output.shape), np.prod(self.kernel_size)),
                              dtype=tensor.Tensor.DEFAULT_DTYPE).T
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
    