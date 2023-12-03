import numpy as np
from typing import Tuple

from .. import functions as F
from ..tensor import Tensor
from .module import Module
from .linear import Linear
from .dropout import Dropout


class GroupedQueryRotaryAttention(Module):
    """Grouped-query rotary attention with KV caching.
    
    This is a generalization of MultiheadAttention. In particular, 
        MultiheadAttention = GroupedQueryRotaryAttention(num_groups=num_heads,
                                                         apply_rotary_embedding=False)
                                                         
    References
    ----------
    [1] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai.
        GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv:2305.13245, 2023
    [2] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
        RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021
                                                         
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: float,
                 num_groups: int,
                 apply_rotary_embedding: bool,
                 rotary_base: int = 10000,
                 max_seqlen: int = 4096):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_groups == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dims_per_head = embed_dim // num_heads
        self.num_groups = num_groups
        self.apply_rotary_embedding = apply_rotary_embedding
        
        self.attention = DotProductAttention(dropout_p)
        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, num_groups * self.dims_per_head)
        self.W_v = Linear(embed_dim, num_groups * self.dims_per_head)
        self.W_o = Linear(embed_dim, embed_dim)
        
        # kv_cache == (key_cache, value_cache), both of shape (batch, num_groups, cache_seqlen, dims_per_head).
        self.kv_cache: Tuple[Tensor] = None
            
        if apply_rotary_embedding:
            # Precompute the sparse rotation matrix for rotary embedding
            # angle: shape (max_seqlen, dims_per_head/2)
            #        angle[i, j] = i / rotary_base^(2j / embed_dim)
            # cos_A, sin_A: shape (max_seqlen, dims_per_head/2, 2)
            angle = np.outer(
                np.arange(max_seqlen),
                rotary_base ** (2 * np.arange(self.dims_per_head // 2) / self.dims_per_head)
            )
            self.cos_A = Tensor(np.stack([np.cos(angle), np.cos(angle)], axis=2))
            self.sin_A = Tensor(np.stack([np.sin(angle), -np.sin(angle)], axis=2))

        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor = None,
                use_kv_cache: bool = False):
        """Attention aggregation.
        
        Parameters
        ----------
        query
            Tensor of shape (batch, query seqlen, embed_dim).
        key, value
            Tensors of shape (batch, key seqlen, embed_dim).
        attn_mask
            Tensor of shape (query seqlen, key seqlen). attn_mask[i, j] = 0 means that query_i
                should attend to key_j.
            If use_kv_cache is True, then attn_mask will be augmented attn_mask with 0s to shape 
                (query seqlen, cache seqlen + key seqlen), which assumes that the query attends 
                to all keys/values in the kv cache.
        use_kv_cache
            Whether or not to use kv_cache. If True, then prepends self.kv_cache to key/value
            before computing attention, and updates self.kv_cache with the new key/value.
            
        Returns
        -------
        (attn_output, attn_scores)
            attn_output is shape (batch, query seqlen, embed_dim)
            attn_scores is shape (batch, query seqlen, key seqlen)
                or (batch, query seqlen, cache seqlen + key seqlen) if use_kv_cache is True
            
        """
        def reshape_and_transpose(tensor, N):
            """Reshapes tensor with shape (batch, seqlen, N * dims_per_head)
                                 to shape (batch, N, seqlen, dims_per_head)."""
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], N, self.dims_per_head))
            tensor = tensor.transpose(1, 2)
            return tensor

        def inv_reshape_and_transpose(tensor):
            """The inverse of reshape_and_transpose."""
            tensor = tensor.transpose(1, 2)
            tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], -1))
            return tensor

        def apply_rotation_matrix(qk, position_offset: int):
            """For each 2D point (x, y) at position index `i` and embed_dim index `(2j, 2j+1)`,
            rotate (x, y) by angle A := i / rotary_base^(2j / embed_dim).

            This means (x, y) -> (cos(A) * x + sin(A) * y, cos(A) * y - sin(A) * x)
                               = (cos(A), cos(A)) o (x, y) + (sin(A), -sin(A)) o (y, x)  .
                                 where o is the elementwise product
                                 
            If `position_offset` is provided, adds `position_offset` to `i`.
            """
            # Reshape to (batch, head, seqlen, dims_per_head/2, 2)
            qk = qk.reshape((*qk.shape[:3], -1, 2))

            seqlen = qk.shape[2]
            
            # Compute (cos(A), cos(A)) o (x, y) + (sin(A), -sin(A)) o (y, x)
            qk_rotated = (self.cos_A[position_offset:position_offset + seqlen] * qk
                          + self.sin_A[position_offset:position_offset + seqlen] * F.flip(qk, axis=-1))

            # Reshape back to (batch, head, seqlen, dims_per_head)
            qk_rotated = qk_rotated.reshape((*qk_rotated.shape[:3], -1))

            return qk_rotated

        query = reshape_and_transpose(self.W_q(query), N=self.num_heads)
        key = reshape_and_transpose(self.W_k(key), N=self.num_groups)
        value = reshape_and_transpose(self.W_v(value), N=self.num_groups)
    
        if self.apply_rotary_embedding:
            position_offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
            query = apply_rotation_matrix(query, position_offset)
            key = apply_rotation_matrix(key, position_offset)

        if use_kv_cache:
            if self.training:
                raise RuntimeError('GroupedQueryRotaryAttention must be in .eval() mode if using KV caching.')

            if self.kv_cache is not None:
                (key_cache, value_cache) = self.kv_cache

                # Prepend key_cache, value_cache along seqlen dimension
                key = F.concat([key_cache, key], axis=2)
                value = F.concat([value_cache, value], axis=2)

                # Augments attn_mask with 0s to shape (query seqlen, cache seqlen + key seqlen),
                # which assumes that the query attends to all keys/values in the kv cache.
                if attn_mask is not None:
                    cache_attn_mask = Tensor(np.zeros((len(attn_mask), self.get_kv_cache_seqlen())))
                    attn_mask = F.concat([cache_attn_mask, attn_mask], axis=1)

            self.kv_cache = (key, value)
            
        # Repeat key and value along head dimension
        if self.num_heads != self.num_groups:
            key = key.repeat_interleave(repeats=self.num_heads // self.num_groups, axis=1)
            value = value.repeat_interleave(repeats=self.num_heads // self.num_groups, axis=1)

        (attn_output, attn_scores) = self.attention(query, key, value, attn_mask)
        attn_output = inv_reshape_and_transpose(attn_output)
        attn_output = self.W_o(attn_output)

        attn_scores = attn_scores.mean(axis=1)  # Average attention scores across head

        return (attn_output, attn_scores)
    
    
    def get_kv_cache_seqlen(self):
        """Gets sequence length of kv_cache."""
        if self.kv_cache is None:
            return 0
        else:
            # kv_cache[0] and [1] are shape (batch, num_heads, cache_seqlen, dims_per_head)
            return self.kv_cache[0].shape[2]
    
    
    def clear_kv_cache(self):
        """Clears kv_cache."""
        self.kv_cache = None
        
        
    def __repr__(self):
        return (f'GroupedQueryRotaryAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads}, num_groups={self.num_groups}, '
                f'apply_rotary_embedding={self.apply_rotary_embedding})')

    
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
        
        # kv_cache == (key_cache, value_cache), both of shape (batch, num_heads, cache_seqlen, dims_per_head).
        self.kv_cache: Tuple[Tensor] = None
        
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                attn_mask: Tensor = None,
                use_kv_cache: bool = False):
        """Attention aggregation.
        
        Parameters
        ----------
        query
            Tensor of shape (batch, query seqlen, embed_dim).
        key, value
            Tensors of shape (batch, key seqlen, embed_dim).
        attn_mask
            Tensor of shape (query seqlen, key seqlen). attn_mask[i, j] = 0 means that query_i
                should attend to key_j.
            If use_kv_cache is True, then attn_mask will be augmented attn_mask with 0s to shape 
                (query seqlen, cache seqlen + key seqlen), which assumes that the query attends 
                to all keys/values in the kv cache.
        use_kv_cache
            Whether or not to use kv_cache. If True, then prepends self.kv_cache to key/value
            before computing attention, and updates self.kv_cache with the new key/value.
            
        Returns
        -------
        (attn_output, attn_scores)
            attn_output is shape (batch, query seqlen, embed_dim)
            attn_scores is shape (batch, query seqlen, key seqlen)
                or (batch, query seqlen, cache seqlen + key seqlen) if use_kv_cache is True
            
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

        if use_kv_cache:
            if self.training:
                raise RuntimeError('MultiheadAttention must be in .eval() mode if using KV caching.')

            if self.kv_cache is not None:
                (key_cache, value_cache) = self.kv_cache

                # Prepend key_cache, value_cache along seqlen dimension
                key = F.concat([key_cache, key], axis=2)
                value = F.concat([value_cache, value], axis=2)

                # Augments attn_mask with 0s to shape (query seqlen, cache seqlen + key seqlen),
                # which assumes that the query attends to all keys/values in the kv cache.
                if attn_mask is not None:
                    cache_attn_mask = Tensor(np.zeros((len(attn_mask), self.get_kv_cache_seqlen())))
                    attn_mask = F.concat([cache_attn_mask, attn_mask], axis=1)

            self.kv_cache = (key, value)

        (attn_output, attn_scores) = self.attention(query, key, value, attn_mask)
        attn_output = inv_reshape_and_transpose(attn_output)
        attn_output = self.W_o(attn_output)

        attn_scores = attn_scores.mean(axis=1)  # Average attention scores across head

        return (attn_output, attn_scores)
    
    
    def get_kv_cache_seqlen(self):
        """Gets sequence length of kv_cache."""
        if self.kv_cache is None:
            return 0
        else:
            # kv_cache[0] and [1] are shape (batch, num_heads, cache_seqlen, dims_per_head)
            return self.kv_cache[0].shape[2]
    
    
    def clear_kv_cache(self):
        """Clears kv_cache."""
        self.kv_cache = None
        
        
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
