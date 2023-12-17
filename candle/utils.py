"""Various utils."""

import pandas as pd
import numpy as np
import os
import requests
from IPython.display import display_html


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


def softmax(array: np.array):
    """Applies softmax along the last axis of a numpy array."""
    softmax = array - array.max(axis=-1, keepdims=True)  # For numerical stabiility
    softmax = np.power(np.e, softmax)
    softmax = softmax / softmax.sum(axis=-1, keepdims=True)
    
    return softmax


def log_softmax(array: np.array):
    """Applies log softmax along the last axis of a numpy array."""
    array = array - array.max(axis=-1, keepdims=True)  # For numerical stabiility
    e_array = np.power(np.e, array)

    log_softmax = array - np.log(e_array.sum(axis=-1, keepdims=True))

    return log_softmax


def conv2d(image: np.array,
           kernel: np.array,
           padding: int = 0,
           stride: int = 1,
           dtype = None):
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
                                image.shape[3] - kernel.shape[3] + 1),  # width
                               dtype=dtype)

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

    if not os.path.isfile(cache_file_path):
        print(f'Downloading from {url} and caching to {cache_file_name}')
        contents = requests.get(url).content
        with open(cache_file_path, 'wb') as f:
            f.write(contents)
    else:
        print(f'Loading file from cache: {cache_file_path}')

    with open(cache_file_path, 'r', encoding=encoding) as f:
        return f.read()


def display_sbs(*args, margin: int = 20):
    html_str = ''
    for df in args:
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += df.to_html().replace('table', f'table style="display:inline;margin-left:{margin}px"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)
    