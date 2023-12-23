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


class GPT(Module):
    
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 dropout_p: float):
        super().__init__()
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.dropout = candle.Dropout(dropout_p)
        self.word_embeddings = candle.Embedding(vocab_size, embed_dim)
        self.position_embeddings = candle.Embedding(block_size, embed_dim)
        self.decoder_blocks = candle.ParameterList([DecoderBlock(embed_dim, n_heads, dropout_p)
                                                    for _ in range(n_layers)])
        self.layer_norm = candle.LayerNorm(embed_dim)
        
        # Tie output projection weights to word embeddings. See "Weight Tying" paper.
        self.output_projection = self.word_embeddings.embeddings
        
    
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
    

    @staticmethod
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
        from .loadpretrained import load_pretrained_gpt
        return load_pretrained_gpt(model_name)
    
    
    def clear_kv_cache(self):
        """Clears kv_cache."""
        for decoder_block in self.decoder_blocks:
            decoder_block.attn.clear_kv_cache()
            
            
    def get_kv_cache_seqlen(self):
        """Gets KV cache seqlen."""
        return self.decoder_blocks[0].attn.get_kv_cache_seqlen()
    
    
    def init_weights(self):
        """Initialize weights for training."""
        params = self.parameters()
        for name in params:
            # Initialize all biases to 0
            if name.endswith('.b'):
                candle.init.zeros_(params[name])
        
            # Initialize layers to N(0.0, 0.02) as per GPT2 paper
            if (
                ('attn' in name or 'ffn' in name) and (name.endswith('.W'))  # Attn/FFN layers
                or ('embeddings' in name) or ('output_projection' in name)
            ):
                candle.init.normal_(params[name], std=0.02)
        
            # Initialize residual outputs as N(0.0, 0.02 / sqrt(2 * n_layers))
            # The intuition is roughly, following similar reasoning to "Improving Transformer Optimization Through Better 
            # Initialization, Huang et al. 2020" you want the gradient norm to be independent of depth. There are 2 "updates"
            # to the residual stream per layer (one from attn and one from FFN), and so you want each update to have variance
            # proportional to 1/(2*n_layers) to keep total variance independent of depth.
            if name.endswith('attn.W_o.W') or name.endswith('ffn.linear2.W'):
                candle.init.normal_(params[name], std=0.02 / np.sqrt(2 * self.n_layers))
            
    
class DecoderBlock(Module):
    
    def __init__(self,
                 embed_dim: int,
                 n_heads: int,
                 dropout_p: float):
        super().__init__()
        self.dropout = candle.Dropout(dropout_p)
        
        self.ln1 = candle.LayerNorm(embed_dim)
        self.attn = candle.MultiheadAttention(embed_dim, n_heads, dropout_p, batch_first=True)
        self.ln2 = candle.LayerNorm(embed_dim)
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
    