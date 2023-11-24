"""Transformations on images, mostly for data augmentation."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from ..tensor import Tensor


class ImageTransform(ABC):
    
    @abstractmethod
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        pass

    
class Compose(ImageTransform):
    
    def __init__(self,
                 transforms: List[ImageTransform]):
        """Chains together multiple transforms into a single transform.

        Parameters
        ----------
        transforms
            Transforms to chain.

        """
        self.transforms = transforms
    
    
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        for transform in self.transforms:
            images = transform(images)
            
        return images


class RandomCrop(ImageTransform):
    
    def __init__(self,
                 size: Union[int, Tuple[int, int]],
                 padding: int = 0):
        """Randomly crop images at random location.

        Parameters
        ----------
        size
            Size of cropped output image.
            If int, then represents (size, size).
            If tuple of ints, then represents (height, width).
        padding
            Padding to use equally on left/right/top/bottom.

        """
        if type(size) is int:
            size = (size, size)
        self.size = size
        self.padding = padding
    
    
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        images = np.pad(images.data, pad_width=[(0, 0),
                                                (0, 0),
                                                (self.padding, self.padding),
                                                (self.padding, self.padding)])

        random_h = np.random.randint(0, images.shape[2] - self.size[0] + 1)
        random_w = np.random.randint(0, images.shape[3] - self.size[1] + 1)

        images = images[:, :, random_h: random_h + self.size[0], random_w: random_w + self.size[1]]

        return Tensor(images)
    
    
class RandomHorizontalFlip(ImageTransform):
    
    def __init__(self,
                 p: float = 0.5):
        """Randomly flips image horizontally.

        Parameters
        ----------
        p
            Probability of image being flipped.

        """
        self.p = p
    
    
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        if np.random.random() < self.p:
            return Tensor(images.data[:, :, :, ::-1])
        else:
            return images

        
class RandomVerticalFlip(ImageTransform):
    
    def __init__(self,
                 p: float = 0.5):
        """Randomly flips image vertically.

        Parameters
        ----------
        p
            Probability of image being flipped.

        """
        self.p = p
    
    
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        if np.random.random() < self.p:
            return Tensor(images.data[:, :, ::-1, :])
        else:
            return images
        
        
class Normalize(ImageTransform):
    
    def __init__(self,
                 means: List[float],
                 stds: List[float]):
        """Normalizes an image using a given mean and std.

        Parameters
        ----------
        means, stds
            Length `channel` list of means/stds per channel.

        """
        if isinstance(means, (int, float)):
            means = [means]
        if isinstance(stds, (int, float)):
            stds = [stds]
        self.means = np.array(means)[None, :, None, None]
        self.stds = np.array(stds)[None, :, None, None]
    
    
    def __call__(self, images):
        """Apply transform.

        Parameters
        ----------
        images
            Tensor of shape (batch, channel, height, width)

        """
        if images.shape[1] != self.means.shape[1]:
            raise ValueError(f'Channel dimension of images is {images.shape[1]}, but channel '
                             f'dimension of normalize transform is {self.means.shape[1]}.')
            
        return Tensor((images.data - self.means) / self.stds)
