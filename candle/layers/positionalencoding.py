import numpy as np

from .. import functions as F
from ..tensor import Tensor
from .module import Module
from .dropout import Dropout


class PositionalEncoding(Module):
    """Fixed positional encoding used in Transformers."""
    
    def __init__(self,
                 embed_dim: int,
                 dropout_p: float,
                 max_len: int = 1000):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.dropout = Dropout(dropout_p)
        
        numer = np.arange(max_len)[:, None]  # shape (max_len, 1)
        denom = 10000 ** (2 * np.arange(embed_dim // 2) / embed_dim)[None, :]  # shape (1, embed_dim)

        self.encoding = Tensor(np.empty((max_len, embed_dim)))
        self.encoding.data[:, ::2] = np.sin(numer / denom)
        self.encoding.data[:, 1::2] = np.cos(numer / denom)
        
        
    def forward(self, x):
        """Adds fixed positional encoding.
        
        Parameters
        ----------
        x
            Tensor of shape (batch, seqlen, embed_dim)
            
        """
        x = x + self.encoding[:x.shape[1], :]
        x = self.dropout(x)
            
        return x
    
        
    def __repr__(self):
        return (f'PositionalEncoding(embed_dim={self.embed_dim})')