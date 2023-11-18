import numpy as np

from ..tensor import Tensor
from ..parameter import Parameter
from .module import Module

    
class Embedding(Module):
    
    def __init__(self,
                 num_embed: int,
                 embed_dim: int):
        self.embeddings = Parameter(Tensor(np.random.normal(size=(num_embed, embed_dim))))
        
        
    def forward(self, indices):
        """Returns embeddings associated with indices.
        
        Parameters
        ----------
        indices
            Integer tensor.
            
        """
        return self.embeddings[list(x.data.astype(int))]
    