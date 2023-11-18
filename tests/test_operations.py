import numpy as np
import unittest

from candle.tensor import Tensor
from candle.operations import (
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Power,
    TensorSum,
    TensorContraction,
    TensorSlice,
    TensorTranspose,
    BatchMatrixMultiply,
    Conv2dOperation,
    MaxPool2dOperation,
    AvgPool2dOperation,
    ReLUActivation,
)
from .utils import numerical_grad_check


class TestOperations(unittest.TestCase):
    
    def test_add_sub_mult_div(self):
        numerical_grad_check(operation_class=Addition,
                             test_inputs=[Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                                          Tensor(np.random.normal(size=(2, 3, 5, 7, 11)))])

        numerical_grad_check(operation_class=Addition,
                             test_inputs=[Tensor(5.0),
                                          Tensor(3.0)])
        
        numerical_grad_check(operation_class=Subtraction,
                             test_inputs=[Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                                          Tensor(np.random.normal(size=(2, 3, 5, 7, 11)))])

        numerical_grad_check(operation_class=Subtraction,
                             test_inputs=[Tensor(5.0),
                                          Tensor(3.0)])

        numerical_grad_check(operation_class=Multiplication,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ])

        numerical_grad_check(operation_class=Multiplication,
                             test_inputs=[
                                 Tensor(5.0),
                                 Tensor(3.0),
                             ])

        numerical_grad_check(operation_class=Division,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                                 Tensor(10 + np.random.normal(size=(2, 3, 5, 7, 11))),
                             ])

        numerical_grad_check(operation_class=Division,
                             test_inputs=[
                                 Tensor(5.0),
                                 Tensor(3.0),
                             ])
        
    def test_broadcasted_add_sub_mult_div(self):
        numerical_grad_check(operation_class=Addition,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(np.random.normal(size=(3, 1, 1, 7, 1))),
                             ])

        numerical_grad_check(operation_class=Addition,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(3.0),
                             ])

        numerical_grad_check(operation_class=Subtraction,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(np.random.normal(size=(3, 1, 1, 7, 1))),
                             ])

        numerical_grad_check(operation_class=Subtraction,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(3.0),
                             ])

        numerical_grad_check(operation_class=Multiplication,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(np.random.normal(size=(3, 1, 1, 7, 1))),
                             ])

        numerical_grad_check(operation_class=Multiplication,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(3.0),
                             ])

        numerical_grad_check(operation_class=Division,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(10 + np.random.normal(size=(3, 1, 1, 7, 1))),
                             ])

        numerical_grad_check(operation_class=Division,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                                 Tensor(3.0),
                             ])

        
    def test_activations(self):
        numerical_grad_check(operation_class=ReLUActivation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ])
        
        
    def test_power(self):
        numerical_grad_check(operation_class=Power,
                             test_inputs=[
                                 Tensor(100 + np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'power': 1.234})

        numerical_grad_check(operation_class=Power,
                             test_inputs=[
                                 Tensor(1e-2 * np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'power': 3})
        
    def test_tensor_contraction(self):
        numerical_grad_check(operation_class=TensorContraction,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                                 Tensor(np.random.normal(size=(5, 7, 11, 17)))
                             ],
                             kwargs = {'axes': 3})

        numerical_grad_check(operation_class=TensorContraction,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(1, 2, 3, 4, 5))),
                                 Tensor(np.random.normal(size=(1, 2, 3, 4, 5)))
                             ],
                             kwargs = {'axes': 5})
        
        
    def test_tensor_sum(self):
        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ])

        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'axis': 1})

        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'axis': (2, 3)})

        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'keepdims': True})

        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'axis': 1,
                                     'keepdims': True})

        numerical_grad_check(operation_class=TensorSum,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(2, 3, 5, 7, 11))),
                             ],
                             kwargs={'axis': (2, 3),
                                     'keepdims': True})
    def test_tensor_slice(self):
        numerical_grad_check(operation_class=TensorSlice,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                             ],
                             kwargs={'key': (slice(None, 3, None), 1, 2, slice(None, None, None))})
        
        
    def test_tensor_transpose(self):
        numerical_grad_check(operation_class=TensorTranspose,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(7, 2, 3, 5, 1, 1, 1))),
                             ],
                             kwargs={'dim0': 2, 'dim1': 5})
        
        
    def test_batch_matrix_multiply(self):
        numerical_grad_check(operation_class=BatchMatrixMultiply,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(3, 4, 5, 7, 11))),
                                 Tensor(np.random.normal(size=(3, 4, 5, 11, 13))),
                             ])
        
    
    def test_conv(self):
        numerical_grad_check(operation_class=Conv2dOperation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(5, 3, 13, 17))),
                                 Tensor(np.random.normal(size=(3, 2, 4, 5))),
                             ],
                             kwargs={'stride': (2, 3),
                                     'padding': (3, 4)})
        
    def test_pool(self):
        numerical_grad_check(operation_class=MaxPool2dOperation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(5, 3, 13, 17))),
                             ],
                             kwargs={'kernel_size': (3, 4),
                                     'padding': (1, 2)})

        numerical_grad_check(operation_class=MaxPool2dOperation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(5, 3, 13, 17))),
                             ],
                             kwargs={'kernel_size': (3, 4),
                                     'padding': (1, 1)})

        numerical_grad_check(operation_class=AvgPool2dOperation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(5, 3, 13, 17))),
                             ],
                             kwargs={'kernel_size': (3, 4),
                                     'padding': (1, 2)})

        numerical_grad_check(operation_class=AvgPool2dOperation,
                             test_inputs=[
                                 Tensor(np.random.normal(size=(5, 3, 13, 17))),
                             ],
                             kwargs={'kernel_size': (3, 4),
                                     'padding': (1, 1)})
