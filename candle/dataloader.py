"""Lightweight data loader."""

import numpy as np
from typing import List, Callable

from .tensor import Tensor


class DataLoader:
    """Data loader for loading Tensors."""
    
    def __init__(self,
                 *tensors: List[Tensor],
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False,
                 transforms: List[Callable] = None):
        """Initialize data loader.
        
        Args:
            tensors (List[Tensor]): Tensors to sample batches from.
            batch_size (int): Size of batches to return.
            shuffle (bool): False to return batches in order.
            drop_last (bool): True to drop the last batch if len(tensors) isn't evenly divisible by batch size.
            transforms (List[Callable]): List with same size as tensors.
                Each transforms[i] is a Callable functions. transforms[i] will apply on tensors[i].
        
        """
        if not len(set([len(x) for x in tensors])) == 1:
            raise ValueError('Tensors must all have the same length.')
            
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transforms = transforms
        
        
    def __len__(self):
        return int(np.ceil(len(self.tensors[0]) / self.batch_size))
        
        
    def __iter__(self):
        return DataLoaderIterator(*self.tensors,
                                  batch_size=self.batch_size, 
                                  shuffle=self.shuffle,
                                  drop_last=self.drop_last,
                                  transforms=self.transforms)

    
class DataLoaderIterator:
    
    def __init__(self,
                 *tensors: List[Tensor],
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 transforms: List[Callable]):
        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.transforms = transforms
        
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
            
            items = []
            for (i, tensor) in enumerate(self.tensors):
                item = tensor[indices]
                
                if self.transforms is not None and self.transforms[i] is not None:
                    item = self.transforms[i](item)
                        
                items.append(item)
            items = tuple(items)
            
            self.index += self.batch_size
            
            return items


class TokenDataLoader:
    """Data loader for loading sequences of integer tokens."""
    
    def __init__(self,
                 sentences: List[List[int]],
                 batch_size: int,
                 pad_token: int,
                 truncate_len: int,
                 group_by_len: bool = True,
                 drop_last: bool = False,):
        """Initialize token data loader.
        
        Args:
            sentences (List[List[int]]): List of encoded sentences.
            batch_size (int): Size of batches to return.
            pad_token (int): Token to pad with.
            truncate_len (int): Length to truncate tokens at.
            group_by_len (bool): Group together samples of roughly the same length.
        
        """
        if not group_by_len:
            raise ValueError('group_by_len = False is not supported.')
            
        self.sentences = sentences
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.truncate_len = truncate_len
        self.group_by_len = group_by_len
        self.drop_last = drop_last
        
        
    def __len__(self):
        return int(np.ceil(len(self.sentences) / self.batch_size))
        
        
    def __iter__(self):
        return TokenDataLoaderIterator(sentences=self.sentences,
                                       batch_size=self.batch_size, 
                                       pad_token=self.pad_token,
                                       truncate_len=self.truncate_len,
                                       group_by_len=self.group_by_len,
                                       drop_last=self.drop_last)

    
class TokenDataLoaderIterator:
    
    def __init__(self,
                 sentences: List[List[int]],
                 batch_size: int,
                 pad_token: int,
                 truncate_len: int,
                 group_by_len: bool,
                 drop_last: bool):
        self.sentences = [i[:truncate_len] for i in sentences]
        self.batch_size = batch_size
        self.pad_token = pad_token
        self.truncate_len = truncate_len
        self.group_by_len = group_by_len
        self.drop_last = drop_last
        
        
    def __next__(self):
        if self.drop_last:
            num_items_required = self.batch_size
        else:
            num_items_required = 1
            
        if len(self.sentences) < num_items_required:
            raise StopIteration
        
        # Grab random sentence as the first element of the batch
        random_sentence_i = np.random.choice(len(self.sentences), 1)[0]
        random_sentence = self.sentences.pop(random_sentence_i)
        batch = [random_sentence]
        remaining_batch_size = min(self.batch_size - 1, len(self.sentences))
        
        # Get 2*remaining_batch_size candidate sentences with closest length to the random sentence
        # Then, choose remaining_batch_size of them
        if remaining_batch_size > 0:
            distances = [abs(len(sentence) - len(random_sentence)) for sentence in self.sentences]
            candidates = np.argsort(distances)[:2 * remaining_batch_size]
            rest_of_batch = np.random.choice(candidates, remaining_batch_size, replace=False)
            batch += [self.sentences.pop(i) for i in sorted(rest_of_batch)[::-1]]
        
            # Pad with padding token
            max_len = max([len(i) for i in batch])
            batch = [i + [self.pad_token] * (max_len - len(i)) for i in batch]

        batch = Tensor(batch).astype(np.int32)
        
        return batch
