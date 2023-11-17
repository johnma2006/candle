import numpy as np
from numpy.random import RandomState
from typing import List, Type, Callable

from candle.tensor import Tensor


def numerical_grad_check(operation_class: Type,
                         test_inputs: List[Tensor],
                         kwargs: dict = {},
                         eps: float = 1e-6,
                         random_seed: int = None):
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
        
    """
    operation = operation_class(test_inputs, **kwargs)
    output = operation.forward()

    # Generate random output_grad and compute input_grads

    output_grad = RandomState(random_seed).normal(size=output.shape)
    input_grads = operation.backward(output_grad)
    loss = (output * output_grad).sum()

    # Numerically estimate input_grads

    numerical_input_grads = []
    for input_i in range(len(test_inputs)):
        test_inputs_eps = test_inputs.copy()
        flattened_input = test_inputs[input_i].data.flatten()

        numerical_input_grad = []
        for i in range(len(flattened_input)):
            flattened_input[i] += eps
            test_inputs_eps[input_i] = Tensor(flattened_input.reshape(test_inputs[input_i].shape))

            output_eps = operation_class(test_inputs_eps, **kwargs).forward()
            loss_eps = (output_eps * output_grad).sum()

            numerical_input_grad.append((loss_eps - loss).data / eps)

            flattened_input[i] -= eps

        numerical_input_grad = np.array(numerical_input_grad).reshape(test_inputs[input_i].shape)
        numerical_input_grads.append(numerical_input_grad)

    # Compare input_grads and numerical_input_grads

    for (i, (input_grad, numerical_input_grad)) in enumerate(zip(input_grads, numerical_input_grads)):
        if np.isclose(input_grad.std(), 0.0) and np.isclose(numerical_input_grad.std(), 0.0):
            corr = 1.0
        else:
            corr = np.corrcoef(input_grad.flatten(), numerical_input_grad.flatten())[0, 1]
        max_diff = np.abs(input_grad - numerical_input_grad).max()

        print(f'input_grads[{i}] vs. numerical_input_grads[{i}]: '
              + f'corr = {100 * corr:.2f}%, '.ljust(17)
              + f'max(abs(diff)) = {max_diff}')
        
        assert corr > 1 - 1e-8
        assert max_diff < 1e-5

        
def model_numerical_grad_check(model,
                               loss_fn: Callable,
                               eps: float = 1e-6,
                               random_seed: int = 123):
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
            flattened_data[i] += eps
            parameter.data = flattened_data.reshape(parameter.shape)
            loss_eps = loss_fn(model, random_seed=random_seed)
            numerical_param_grad.append((loss_eps - loss).data / eps)
            flattened_data[i] -= eps

        parameter.data = flattened_data.reshape(parameter.shape)
        numerical_param_grad = np.array(numerical_param_grad).reshape(parameter.shape)
        numerical_param_grads.append(numerical_param_grad)

    # Compare param_grads and numerical_param_grads

    for (param_name, param_grad, numerical_param_grad) in zip(parameters.keys(), param_grads, numerical_param_grads):
        if np.isclose(param_grad.std(), 0.0) and np.isclose(numerical_param_grad.std(), 0.0):
            corr = 1.0
        else:
            corr = np.corrcoef(param_grad.flatten(), numerical_param_grad.flatten())[0, 1]
        max_diff = np.abs(param_grad - numerical_param_grad).max()

        print(f'{param_name}: '.ljust(30)
              + f'corr = {100 * corr:.2f}%, '.ljust(17)
              + f'max(abs(diff)) = {max_diff}')
        
        assert corr > 1 - 1e-8
        assert max_diff < 1e-5
        