"""Weight initialization."""

import numpy as np


def normal_(tensor, mean: float = 0.0, std: float = 1.0):
    weight = np.random.normal(loc=mean, scale=std, size=tensor.shape)
    tensor.data[:] = weight


def uniform_(tensor, a: float = 0.0, b: float = 1.0):
    weight = np.random.uniform(low=a, high=b, size=tensor.shape)
    tensor.data[:] = weight


def zeros_(tensor):
    tensor.data[:] = 0


def xavier_(tensor):
    tensor.data[:] = xavier(tensor.shape)


def kaiming_(tensor):
    tensor.data[:] = kaiming(tensor.shape)


def xavier(shape) -> np.ndarray:
    (in_shape, out_shape) = shape
    weight = np.random.uniform(low=-np.sqrt(6 / (in_shape + out_shape)),
                               high=np.sqrt(6 / (in_shape + out_shape)),
                               size=(in_shape, out_shape))
    return weight


def kaiming(shape) -> np.ndarray:
    (in_shape, out_shape) = shape
    weight = np.random.uniform(low=-np.sqrt(6 / in_shape),
                               high=np.sqrt(6 / in_shape),
                               size=(in_shape, out_shape))
    return weight
