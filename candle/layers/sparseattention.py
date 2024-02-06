import numpy as np
from typing import Tuple

from .. import functions as F
from ..tensor import Tensor
from .module import Module


class FixedSparseAttention(Module):
    """Fixed sparse autoregressive attention."""
    
    def __init__(self,
                 block_size: int):
        super().__init__()
        self.block_size = block_size
        
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor):
        """Does attention aggregation.
        
        Args:
            query, key, value (Tensor): shape (batch, seqlen, embed_dim)
                seqlen must be divisible by block_size.
            
        Returns:
            Tensor: attn_output with shape (batch, seqlen, embed_dim)
            
        """
        (batch, seqlen, embed_dim) = query.shape
        n_blocks = seqlen // self.block_size

        # shape (batch, n_blocks, block_size, embed_dim)
        query = reshape_into_subblocks(query, self.block_size)
        key = reshape_into_subblocks(key, self.block_size)
        value = reshape_into_subblocks(value, self.block_size)

        # Attend to current subblock

        # shape (batch, n_blocks, block_size, block_size)
        logits = F.bmm(query, key.transpose(-1, -2)) / np.sqrt(embed_dim)
        
        causal_attn_mask = Tensor(1 - np.tri(self.block_size))
        logits = F.masked_fill(logits, causal_attn_mask, fill_value=-1e9)

        # Aggregate by subblock
        
        attn_scores = F.softmax(logits)
        attn_output = F.bmm(attn_scores, value)
        
        attn_output = attn_output.reshape((batch, seqlen, embed_dim))
        
        return attn_output
        
        
    def __repr__(self):
        return f'FixedSparseAttention(dropout={self.block_size})'


class StridedSparseAttention(Module):
    """Strided sparse autoregressive attention."""
    
    def __init__(self,
                 block_size: int):
        super().__init__()
        self.block_size = block_size
        
        
    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor):
        """Does attention aggregation.
        
        Args:
            query, key, value (Tensor): shape (batch, seqlen, embed_dim)
                seqlen must be divisible by block_size.
            
        Returns:
            Tensor: attn_output with shape (batch, seqlen, embed_dim)
            
        """
        NEG_INF = -1e9
        (batch, seqlen, embed_dim) = query.shape
        n_blocks = seqlen // self.block_size

        # shape (batch, n_blocks, block_size, embed_dim)
        query = reshape_into_subblocks(query, self.block_size)
        key = reshape_into_subblocks(key, self.block_size)
        value = reshape_into_subblocks(value, self.block_size)
        
        query_prev = query[:, 1:]
        key_prev = key[:, :-1]
        value_prev = value[:, :-1]
        
        # Attend to current subblock

        # shape (batch, n_blocks, block_size, block_size)
        logits = F.bmm(query, key.transpose(-1, -2)) / np.sqrt(embed_dim)
        
        causal_attn_mask = Tensor(1 - np.tri(self.block_size))
        logits = F.masked_fill(logits, causal_attn_mask, fill_value=-1e9)
        
        # Attend to previous subblock

        # shape (batch, n_blocks - 1, block_size, block_size)
        logits_prev = F.bmm(query_prev, key_prev.transpose(-1, -2)) / np.sqrt(embed_dim)
        
        prev_attn_mask = Tensor(np.tri(self.block_size))
        logits_prev = F.masked_fill(logits_prev, prev_attn_mask, fill_value=NEG_INF)

        # pad to shape (batch, n_blocks, block_size, block_size)
        neg_inf = Tensor(NEG_INF * np.ones((batch, 1, self.block_size, self.block_size)))
        logits_prev = F.concat([neg_inf, logits_prev], axis=1)
        
        # Aggregate attention between previous and current subblocks

        # shape (batch, n_blocks, block_size, 2 * block_size)
        logits_prev_and_curr = F.concat([logits_prev, logits], axis=-1)
        attn_scores = F.softmax(logits_prev_and_curr)
        
        (attn_scores_prev, attn_scores) = attn_scores.split([self.block_size, self.block_size],
                                                            axis=-1)
        
        attn_output = F.bmm(attn_scores, value)
        attn_output_prev = F.bmm(attn_scores_prev[:, 1:], value_prev)
        
        attn_output[:, 1:] += attn_output_prev
        attn_output = attn_output.reshape((batch, seqlen, embed_dim))

        return attn_output
        
        
    def __repr__(self):
        return f'StridedSparseAttention(dropout={self.block_size})'


def reshape_into_subblocks(x: Tensor, block_size: int):
    """Transposes a Tensor x into sub-blocks.

    Args:
        x: shape (batch, seqlen, embed_dim)
        block_size: size of sub-blocks.

    Returns
        Tensor: shape (batch, seqlen // subblocks, subblocks, embed_dim)
    
    """
    (batch, seqlen, embed_dim) = x.shape
    
    if seqlen % block_size != 0:
        raise ValueError(f'seqlen dimension = {seqlen} must be divisible by block_size = {block_size}.')

    x = x.reshape((batch, seqlen // block_size, block_size, embed_dim))

    return x
