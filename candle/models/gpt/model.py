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
    
    
    def forward(self, indices):
        """
        Parameters
        ----------
        indices
            Integer tensor with shape (batch, seq_len).
        
        Returns
        -------
        logits
            Tensor with shape (batch, seqlen, vocab_size).
            
        """
        # x: shape (batch, seqlen, embed_dim)
        position_indices = Tensor(np.arange(indices.shape[1]))
        
        x = self.word_embeddings(indices) + self.position_embeddings(position_indices)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

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

        
    def forward(self, x):
        # x: shape (batch, seqlen, embed_dim)
        x = x + self.dropout(self.self_attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        
        return x

    
    def self_attn(self, x):
        """Self-attention with causal mask."""
        # causal_attn_mask[i, j] = 0 means that query[i] attends to key[j], and so
        # causal_attn_mask[i, j] = 0 if i >= j and 1 otherwise.
        causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))
        
        (attn_output, attn_scores) = self.attn(x, x, x, attn_mask=causal_attn_mask)
        
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
    