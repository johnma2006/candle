import numpy as np
from numpy.random import RandomState
from typing import List, Type, Callable

from candle.tensor import Tensor


def numerical_grad_check(operation_class: Type,
                         test_inputs: List[Tensor],
                         kwargs: dict = {},
                         eps: float = 1e-4,
                         random_seed: int = None,
                         atol: float = 1e-4,
                         check_dtype: bool = True):
    """Compares .backward() gradient against numerically computed gradient.
    
    Parameters
    ----------
    operation_class
        Class, e.g. TensorDot.
    test_inputs
        List of Tensor inputs into `operation_class`.
    eps
        Perturbation when computing numerical gradient.
    random_seed
        Random seed to use when generating randomness.
    atol
        Tolerance for errors in assertions.
    check_dtype
        If True, then also checks that the dtype of the gradient matches Tensor.DEFAULT_DTYPE
        
    """
    # Switch to double precision for grad checks
    default_dtype = Tensor.DEFAULT_DTYPE
    Tensor.DEFAULT_DTYPE = np.float64
    test_inputs = [i.astype(Tensor.DEFAULT_DTYPE) for i in test_inputs]

    operation = operation_class(test_inputs, **kwargs)
    output = operation.forward()

    # Generate random output_grad and compute input_grads

    output_grad = RandomState(random_seed).normal(size=output.shape).astype(Tensor.DEFAULT_DTYPE)
    input_grads = operation.backward(output_grad)

    # Numerically estimate input_grads

    numerical_input_grads = []
    for input_i in range(len(test_inputs)):
        test_inputs_eps = test_inputs.copy()
        flattened_input = test_inputs[input_i].data.flatten()

        numerical_input_grad = []
        for i in range(len(flattened_input)):
            orig_val = flattened_input[i]
            flattened_input[i] = orig_val + eps
            test_inputs_eps[input_i] = Tensor(flattened_input.reshape(test_inputs[input_i].shape))
            output_eps = operation_class(test_inputs_eps, **kwargs).forward()
            loss_eps = (output_eps * output_grad)

            flattened_input[i] = orig_val - eps
            test_inputs_eps[input_i] = Tensor(flattened_input.reshape(test_inputs[input_i].shape))
            output_eps = operation_class(test_inputs_eps, **kwargs).forward()
            loss_minus_eps = (output_eps * output_grad)

            numerical_input_grad.append((loss_eps - loss_minus_eps).sum().data / (2 * eps))

            flattened_input[i] = orig_val

        numerical_input_grad = np.array(numerical_input_grad).reshape(test_inputs[input_i].shape)
        numerical_input_grads.append(numerical_input_grad)

    # Compare input_grads and numerical_input_grads

    for (i, (input_grad, numerical_input_grad)) in enumerate(zip(input_grads, numerical_input_grads)):
        if np.isclose(input_grad.std(), 0.0, atol=1e-3) and np.isclose(numerical_input_grad.std(), 0.0, atol=1e-3):
            corr = 1.0
        else:
            corr = np.corrcoef(input_grad.flatten(), numerical_input_grad.flatten())[0, 1]
        numer = np.abs(input_grad - numerical_input_grad)
        denom = np.maximum(np.abs(input_grad), np.abs(numerical_input_grad)) + 1e-5
        relative_err = numer / denom
        if len(relative_err.shape) > 0:
            relative_err[np.isnan(relative_err)] = 0.0
            
        max_relative_err = np.quantile(relative_err, 0.999)  # Not really max but sometimes kinks mess up the gradcheck
        
        class_name = f'[{operation_class.__name__}]'
        print(f'{class_name.ljust(30)} input_grads[{i}] vs. numerical_input_grads[{i}]: '
              + f'corr = {100 * corr:.2f}%, '.ljust(17)
              + f'max(relative_err) = {max_relative_err}')

        assert corr > 1 - 100 * atol, corr
        assert max_relative_err < atol, max_relative_err
        
    Tensor.DEFAULT_DTYPE = default_dtype
    
    # Check dtypes of output_grad
    
    test_inputs = [i.astype(Tensor.DEFAULT_DTYPE) for i in test_inputs]
    
    if check_dtype:
        operation = operation_class(test_inputs, **kwargs)
        output = operation.forward()
        output_grad = RandomState(random_seed).normal(size=output.shape).astype(Tensor.DEFAULT_DTYPE)
        input_grads = operation.backward(output_grad)

        assert output.dtype == Tensor.DEFAULT_DTYPE
        for input_grad in input_grads:
            assert input_grad.dtype == Tensor.DEFAULT_DTYPE

        
def model_numerical_grad_check(model,
                               loss_fn: Callable,
                               eps: float = 1e-5,
                               random_seed: int = 123,
                               atol: float = 1e-3,
                               check_dtype: bool = True):
    """Compares .backward() gradient against numerically computed gradient.
    
    Parameters
    ----------
    model
        Module that implements .forward()
    loss_fn
        Function with signature `def loss_fn(model, random_seed)` that returns loss for some random fixed input.
    eps
        Perturbation when computing numerical gradient.
    random_seed
        Random seed to use when generating randomness.
        
    """
    # Switch to double precision for grad checks
    default_dtype = Tensor.DEFAULT_DTYPE
    Tensor.DEFAULT_DTYPE = np.float64
    
    parameters = model.parameters()

    # Compute grads

    loss = loss_fn(model, random_seed=random_seed)
    loss._reset_graph()
    loss.backward()

    param_grads = [np.array(parameters[key].grad) for key in parameters]

    # Numerically estimate param_grads

    numerical_param_grads = []
    for key in parameters:
        parameter = parameters[key]
        flattened_data = parameter.data.flatten()

        numerical_param_grad = []
        for i in range(len(flattened_data)):
            orig_val = flattened_data[i]
            flattened_data[i] = orig_val + eps
            parameter.data = flattened_data.reshape(parameter.shape)
            loss_eps = loss_fn(model, random_seed=random_seed)
            
            flattened_data[i] = orig_val - eps
            parameter.data = flattened_data.reshape(parameter.shape)
            loss_minus_eps = loss_fn(model, random_seed=random_seed)

            numerical_param_grad.append((loss_eps - loss_minus_eps).data / (2 * eps))
            flattened_data[i] = orig_val

        parameter.data = flattened_data.reshape(parameter.shape)
        numerical_param_grad = np.array(numerical_param_grad).reshape(parameter.shape)
        numerical_param_grads.append(numerical_param_grad)

    # Compare param_grads and numerical_param_grads

    for (param_name, param_grad, numerical_param_grad) in zip(parameters.keys(), param_grads, numerical_param_grads):
        if np.isclose(param_grad.std(), 0.0, atol=1e-3) and np.isclose(numerical_param_grad.std(), 0.0, atol=1e-3):
            corr = 1.0
        else:
            corr = np.corrcoef(param_grad.flatten(), numerical_param_grad.flatten())[0, 1]
        numer = np.abs(param_grad - numerical_param_grad)
        denom = np.maximum(np.abs(param_grad), np.abs(numerical_param_grad)) + 1e-5
        relative_err = numer / denom
        if len(relative_err.shape) > 0:
            relative_err[np.isnan(relative_err)] = 0.0
        max_relative_err = np.quantile(relative_err, 0.999)  # Not really max but sometimes kinks mess up the gradcheck

        print(f'{param_name}: '.ljust(30)
              + f'corr = {100 * corr:.2f}%, '.ljust(17)
              + f'max(relative_err) = {max_relative_err}')
        
        assert corr > 1 - 100 * atol, corr
        assert max_relative_err < atol, max_relative_err
        
    Tensor.DEFAULT_DTYPE = default_dtype
    