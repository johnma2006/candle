import numpy as np

from .. import functions as F
from ..tensor import Tensor
from .module import Module
from .linear import Linear
from .dropout import Dropout

    
class MultiheadAttention(Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: float):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dims_per_head = embed_dim // num_heads
        
        self.attention = DotProductAttention(dropout_p)
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)
        self.W_o = Linear(embed_dim, embed_dim)
        
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor = None):
        """Does attention aggregation.
        
        Parameters
        ----------
        query
            Tensor of shape (batch, query seqlen, embed_dim)
        key, value
            Tensors of shape (batch, key seqlen, embed_dim)
        attn_mask
            Tensor of shape (query seqlen, key seqlen)
            
        """
        def reshape_and_transpose(tensor):
            """Reshapes tensor with shape (batch, seqlen, num_heads * dims_per_head)
                                 to shape (batch, num_heads, seqlen, dims_per_head)."""
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], self.num_heads, self.dims_per_head))
            tensor = tensor.transpose(1, 2)
            return tensor

        def inv_reshape_and_transpose(tensor):
            """The inverse of reshape_and_transpose."""
            tensor = tensor.transpose(1, 2)
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], -1))
            return tensor

        query = reshape_and_transpose(self.W_q(query))
        key = reshape_and_transpose(self.W_k(key))
        value = reshape_and_transpose(self.W_v(value))

        (attn_output, attn_scores) = self.attention(query, key, value, attn_mask)
        attn_output = inv_reshape_and_transpose(attn_output)
        attn_output = self.W_o(attn_output)

        attn_scores = attn_scores.mean(axis=1)  # Average attention scores across head

        return (attn_output, attn_scores)

        
    def __repr__(self):
        return (f'MultiheadAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads})')

    
class DotProductAttention(Module):
    def __init__(self,
                 dropout_p: float):
        super().__init__()
        self.dropout = Dropout(dropout_p)
        
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor = None):
        """Does attention aggregation.
        
        Parameters
        ----------
        query
            Tensor of shape (batch, ..., query seqlen, embed_dim)
        key, value
            Tensors of shape (batch, ..., key seqlen, embed_dim)
        attn_mask
            Tensor of 1s and 0s with shape (query seqlen, key seqlen).
            1 if not allowed to attend, 0 if allowed to attend.
            
        Returns
        -------
        (attn_output, attn_scores)
            attn_output is shape (batch, ..., query seqlen, embed_dim)
            attn_scores is shape (batch, ..., query seqlen, source seqlen)
            
        """
        embed_dim = query.shape[-1]

        # logits: shape (batch, ..., query seqlen, key seqlen)
        logits = F.bmm(query, key.transpose(-1, -2)) / np.sqrt(embed_dim)

        if attn_mask is not None:
            logits = F.masked_fill(logits, attn_mask, fill_value=-1e9)

        attn_scores = F.softmax(logits)
        attn_scores = self.dropout(attn_scores)

        attn_output = F.bmm(attn_scores, value)
        
        return (attn_output, attn_scores)
        
        
    def __repr__(self):
        return f'DotProductAttention(dropout={self.dropout.p})'
    
      
class PositionalEncoding(Module):
    
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
