import numpy as np
from typing import Tuple

from .. import functions as F
from ..tensor import Tensor
from .module import Module
from .linear import Linear
from .dropout import Dropout

    
class MultiheadAttention(Module):
    """Multi-headed attention with KV caching."""
    
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
                attn_mask: Tensor = None,
                kv_cache: Tuple[Tensor, Tensor] = None,
                return_new_kv_cache: bool = False):
        """Does attention aggregation.
        
        Parameters
        ----------
        query
            Tensor of shape (batch, query seqlen, embed_dim)
        key, value
            Tensors of shape (batch, key seqlen, embed_dim)
        attn_mask
            Tensor of shape (query seqlen, key seqlen)
            If kv_cache is provided, attn_mask is shape (query seqlen, key seqlen + cache seqlen)
        kv_cache
            Optional. Tuple of Tensors (key_cache, value_cache), where key_cache and 
            value_cache are both shape (batch, num_heads, cache_seqlen, dims_per_head).
            
            If provided, then pre-pends new_key_cache/new_value_cache to key/value
            before computing attention, and the returns a new_kv_cache as well.
            
            Must be in eval() mode.
        return_new_kv_cache
            If True, then returns the updated kv_cache.
            
        Returns
        -------
        if kv_cache is None:
            (attn_output, attn_scores)
                attn_output is shape (batch, query seqlen, embed_dim)
                attn_scores is shape (batch, query seqlen, key seqlen)
        elif kv_cache is provided:
            (attn_output, attn_scores, new_kv_cache)
                attn_output is shape (batch, query seqlen, embed_dim)
                attn_scores is shape (batch, query seqlen, key seqlen + cache seqlen)
                new_kv_cache = (new_key_cache, new_value_cache) both with
                    shape (batch, num_heads, key seqlen + cache_seqlen, dims_per_head)
            
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

        if kv_cache is not None:
            if self.training:
                raise RuntimeError('MultiheadAttention must be in .eval() mode if using KV caching.')

            (key_cache, value_cache) = kv_cache

            # Prepend key_cache, value_cache along seqlen dimension
            key = F.concat([key_cache, key], axis=2)
            value = F.concat([value_cache, value], axis=2)
            
        (attn_output, attn_scores) = self.attention(query, key, value, attn_mask)
        attn_output = inv_reshape_and_transpose(attn_output)
        attn_output = self.W_o(attn_output)

        attn_scores = attn_scores.mean(axis=1)  # Average attention scores across head

        if return_new_kv_cache:
            new_kv_cache = (key, value)
            return (attn_output, attn_scores, new_kv_cache)
        else:
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
            attn_scores is shape (batch, ..., query seqlen, key seqlen)
            
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
