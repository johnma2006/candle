"""GPT2 implementation.

References:
[1] OpenAI's GPT2: https://github.com/openai/gpt-2
[2] Karpathy's minGPT: https://github.com/karpathy/minGPT

"""

import numpy as np
from typing import List, Tuple

import candle
import candle.functions as F
from candle.tensor import Tensor
from candle.layers.module import Module

from .loadpretrained import load_pretrained_gpt


class GPT(Module):
    
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 dropout_p: float):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.dropout = candle.Dropout(dropout_p)
        self.word_embeddings = candle.Embedding(vocab_size, embed_dim)
        self.position_embeddings = candle.Embedding(block_size, embed_dim)
        self.decoder_blocks = candle.ParameterList([DecoderBlock(embed_dim, num_heads, dropout_p)
                                                    for _ in range(num_layers)])
        self.layer_norm = candle.LayerNorm(axis=2)
        
        # Tie output projection weights to word embeddings. See "Weight Tying" paper.
        self.output_projection = self.word_embeddings.embeddings
        
        # TODO: initialize weights properly
    
    
    def forward(self,
                indices: Tensor,
                use_kv_cache: bool = False):
        """
        Parameters
        ----------
        indices
            Integer tensor with shape (batch, seq_len).
        use_kv_cache
            Whether or not to use kv_cache to speed up inference.
        
        Returns
        -------
        logits
            Tensor with shape (batch, seqlen, vocab_size)
            
        """
        offset = self.get_kv_cache_seqlen() if use_kv_cache else 0
        position_indices = Tensor(np.arange(indices.shape[1]) + offset)
        
        x = self.word_embeddings(indices) + self.position_embeddings(position_indices)
        x = self.dropout(x)  # shape (batch, seqlen, embed_dim)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, use_kv_cache)

        x = self.layer_norm(x)
        
        return x @ self.output_projection.T
    
    
    def from_pretrained(model_name: str):
        """Returns GPT2 with pretrained weights.

        Parameters
        -----------
        model_name
            One of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'].

            Param Count:
                gpt2:        124,439,808
                gpt2-medium: 354,823,168
                gpt2-large:  774,030,080
                gpt2-xl:   1,557,611,200

        Returns
        -------
        model
            GPT instance with pre-trained weights initialized.

        """
        return load_pretrained_gpt(model_name)
    
    
    def clear_kv_cache(self):
        """Clears kv_cache."""
        for decoder_block in self.decoder_blocks:
            decoder_block.attn.clear_kv_cache()
            
            
    def get_kv_cache_seqlen(self):
        """Gets KV cache seqlen."""
        return self.decoder_blocks[0].attn.get_kv_cache_seqlen()
            

    def modify_kv_cache(self,
                        trim_seqlen: int = None,
                        reindex_batch_indices: List[int] = None):
        """Modifies the kv_cache by trimming or reindexing.
        
        Parameters
        ----------
        trim_seqlen
            Trims kv_cache along the seqlen dimension to length `new_seqlen`, keeping later tokens.
            If `trim_seqlen` is negative, then keeps earlier tokens.
        batch_indices
            Reindexes the kv_cache along the batch dimension. Necessary during beam search.
            
        """
        for decoder_block in self.decoder_blocks:
            if decoder_block.attn.kv_cache is not None:
                (key_cache, value_cache) = decoder_block.attn.kv_cache

                if trim_seqlen is not None:
                    if trim_seqlen > 0:
                        key_cache = key_cache[:, :, -trim_seqlen:]
                        value_cache = value_cache[:, :, -trim_seqlen:]
                    elif trim_seqlen < 0:
                        key_cache = key_cache[:, :, :-trim_seqlen]
                        value_cache = value_cache[:, :, :-trim_seqlen]
                    else: # trim_seqlen == 0
                        key_cache = None
                        value_cache = None

                if reindex_batch_indices is not None:
                    key_cache = Tensor(key_cache.data[reindex_batch_indices])
                    value_cache = Tensor(value_cache.data[reindex_batch_indices])

                decoder_block.attn.kv_cache = (key_cache, value_cache)
                
    
class DecoderBlock(Module):
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout_p: float):
        super().__init__()
        self.dropout = candle.Dropout(dropout_p)
        
        self.ln1 = candle.LayerNorm(axis=2)
        self.attn = candle.MultiheadAttention(embed_dim, num_heads, dropout_p)
        self.ln2 = candle.LayerNorm(axis=2)
        self.ffn = FeedForwardBlock(input_dim=embed_dim, hidden_dim=4 * embed_dim)

        
    def forward(self,
                x: Tensor,
                use_kv_cache: bool):
        # x: Tensor with shape (batch, seqlen, embed_dim)
        x = x + self.dropout(self.self_attn(self.ln1(x), use_kv_cache))
        x = x + self.dropout(self.ffn(self.ln2(x)))

        return x

    
    def self_attn(self,
                  x: Tensor,
                  use_kv_cache: bool):
        """Self-attention with causal mask."""
        # causal_attn_mask[i, j] = 0 means that query[i] attends to key[j], and so
        # causal_attn_mask[i, j] = 0 if i >= j and 1 otherwise.
        causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))

        (attn_output, attn_scores) = self.attn(x, x, x,
                                               attn_mask=causal_attn_mask,
                                               use_kv_cache=use_kv_cache)

        return attn_output
    
    
class FeedForwardBlock(Module):
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int):
        super().__init__()
        self.linear1 = candle.Linear(input_dim, hidden_dim)
        self.linear2 = candle.Linear(hidden_dim, input_dim)
        
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        
        return x
    