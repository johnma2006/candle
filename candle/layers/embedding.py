import numpy as np

from ..tensor import Tensor, Parameter
from .module import Module

    
class Embedding(Module):
    
    def __init__(self,
                 num_embed: int,
                 embed_dim: int):
        super().__init__()
        self.embeddings = Parameter(Tensor(np.random.normal(size=(num_embed, embed_dim))))
        
        
    def forward(self, indices):
        """Returns embeddings associated with indices.
        
        Parameters
        ----------
        indices
            Shape (batch, seqlen) integer tensor of embedding indices.
            
        """
        return self.embeddings[list(indices.data.astype(int))]
    