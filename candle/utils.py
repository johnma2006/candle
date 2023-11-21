"""Various utils."""

import numpy as np
import os
import requests


def get_broadcasted_axes(old_shape: tuple,
                         new_shape: tuple):
    """Returns the axes of shape that np.broadcast_to(a, shape) broadcasted to."""
    old_shape = (1,) * (len(new_shape) - len(old_shape)) + old_shape
    broadcasted_axes = tuple([axes for axes in range(len(new_shape)) if old_shape[axes] != new_shape[axes]])
    
    return broadcasted_axes


def sum_along_broadcasted_axes(array, old_shape):
    broadcasted_axes = get_broadcasted_axes(old_shape, array.shape)
    array_sum = array.sum(axis=broadcasted_axes, keepdims=True)
    array_sum = np.array(array_sum)  # Cast floats back to np.array type
    array_sum = array_sum.squeeze(axis=tuple(range(len(array.shape) - len(old_shape))))
    
    return array_sum


def xavier_init(in_shape, out_shape):
    return np.random.uniform(low=-np.sqrt(6 / (in_shape + out_shape)),
                             high=np.sqrt(6 / (in_shape + out_shape)),
                             size=(in_shape, out_shape))


def kaiming_init(in_shape, out_shape):
    return np.random.uniform(low=-np.sqrt(6 / in_shape),
                             high=np.sqrt(6 / in_shape),
                             size=(in_shape, out_shape))


def softmax(array: np.array):
    """Applies softmax along the last axis of a numpy array."""
    softmax = array.T
    softmax = softmax - softmax.max(axis=0)  # For numerical stabiility
    softmax = np.power(np.e, softmax)
    softmax = (softmax / softmax.sum(axis=0)).T
    
    return softmax


def log_softmax(array: np.array):
    """Applies log softmax along the last axis of a numpy array."""
    array = array.T
    array = array - array.max(axis=0)  # For numerical stabiility
    e_array = np.power(np.e, array)

    log_softmax = (array - np.log(e_array.sum(axis=0))).T

    return log_softmax


def conv2d(image: np.array,
           kernel: np.array,
           padding: int = 0,
           stride: int = 1):
    """Computes 2D convolution.

    Parameters
    ----------
    image
        np.array with shape (N, in_channels, height, width).
    kernel
        np.array with shape (in_channels, out_channels, kernel_height, kernel_width).
    padding
        int, or tuple of ints (padding_height, padding_width).
    stride
        int, or tuple of ints (stride_height, stride_width).

    """
    if type(padding) is int:
        padding = (padding, padding)

    if type(stride) is int:
        stride = (stride, stride)

    image = np.pad(image,
                   pad_width=((0, 0),                     # N
                              (0, 0),                     # in_channels
                              (padding[0], padding[0]),   # height
                              (padding[1], padding[1])))  # width

    convolved_image = np.zeros((image.shape[0],                         # N
                                kernel.shape[1],                        # out_channels
                                image.shape[2] - kernel.shape[2] + 1,   # height
                                image.shape[3] - kernel.shape[3] + 1))  # width

    for i in range(kernel.shape[2]):
        for j in range(kernel.shape[3]):
            _image_scaled = np.einsum('jk,ijlm', kernel[:, :, i, j], image, optimize=True)
            convolved_image += _image_scaled[:, :, i:i + convolved_image.shape[2], j:j + convolved_image.shape[3]]

    # Apply stride
    convolved_image = convolved_image[:, :, ::stride[0], ::stride[1]]
    
    return convolved_image
    
    
def download_and_cache_file(url: str, cache_file_name: str, encoding: str = None):
    """Downloads a file from `url` and caches it in {home_dir}/.cache/candle/{cache_file_name}"""
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'candle')
    cache_file_path = os.path.join(cache_dir, cache_file_name)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(cache_file_path):
        with open(cache_file_path, 'r', encoding=encoding) as f:
            return f.read()

    else:
        print(f'Downloading from {url} and caching to {cache_file_name}')
        contents = requests.get(url).content
        with open(cache_file_path, 'wb') as f:
            f.write(contents)
        return contents
    