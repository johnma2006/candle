"""Lightweight data loader."""

import numpy as np
from typing import List

from .tensor import Tensor


class DataLoader:
    
    def __init__(self,
                 *tensors: List[Tensor],
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False):
        if not len(set([len(x) for x in tensors])) == 1:
            raise ValueError('Tensors must all have the same length.')
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        
    def __len__(self):
        return int(np.ceil(len(self.tensors[0]) / self.batch_size))
        
        
    def __iter__(self):
        return DataLoaderIterator(*self.tensors,
                                  batch_size=self.batch_size, 
                                  shuffle=self.shuffle,
                                  drop_last=self.drop_last)

class DataLoaderIterator:
    
    def __init__(self,
                 *tensors: List[Tensor],
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.index = 0
        if shuffle:
            self.ordering = np.random.permutation(len(self.tensors[0]))
        else:
            self.ordering = np.arange(len(self.tensors[0]))
            
        
    def __next__(self):
        num_items_left = len(self.ordering) - self.index
        if self.drop_last:
            num_items_required = self.batch_size
        else:
            num_items_required = 1
            
        if num_items_left < num_items_required:
            raise StopIteration
        else:
            indices = self.ordering[self.index: self.index + self.batch_size]
            items = tuple([tensor[indices] for tensor in self.tensors])
            self.index += self.batch_size
            
            return items
        