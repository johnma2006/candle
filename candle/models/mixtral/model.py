"""Mixtral mixture of experts implementation.

References:
[1] Mistral implementation: https://github.com/mistralai/mistral-src/blob/main/mistral/

"""

import numpy as np
from typing import List, Tuple

import candle
import candle.functions as F
from candle.tensor import Tensor
from candle.layers.module import Module


class Mixtral(Module):
    
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 n_experts: int,
                 n_experts_per_tok: int,
                 vocab_size: int,
                 block_size: int,
                 n_kv_heads: int = None,
                 ffn_hidden_dim: int = None,
                 rotary_base: int = 1000000,
                 norm_eps: float = 1e-5):
        super().__init__()

        if n_kv_heads is None:
            n_kv_heads = n_heads
            
        if ffn_hidden_dim is None:
            # Conventionally, ffn_hidden_dim = 4 * embed_dim
            # Since Mixtral uses the SwiGLU "activation", ffn_hidden_dim is set to 4 * embed_dim * 2/3
            # to keep the number of parameters the same.
            ffn_hidden_dim = int(4 * embed_dim  * (2 / 3))

            # Round ffn_hidden_dim up to the nearest multiple of 256
            if ffn_hidden_dim % 256 != 0:
                ffn_hidden_dim = ffn_hidden_dim + 256 - ffn_hidden_dim % 256
        
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.rotary_base = rotary_base
        self.ffn_hidden_dim = ffn_hidden_dim
        self.norm_eps = norm_eps
        
        self.word_embeddings = candle.Embedding(vocab_size, embed_dim)
        self.decoder_blocks = candle.ParameterList([
            DecoderBlock(embed_dim, ffn_hidden_dim, n_heads, n_kv_heads,
                         n_experts, n_experts_per_tok,
                         block_size, rotary_base, norm_eps)
            for _ in range(n_layers)
        ])
        self.rms_norm = candle.RMSNorm(axis=2, eps=norm_eps)
                
        # Unlike GPT, Mixtral/LLaMA output layer is not tied to word embeddings
        self.output_projection = candle.Linear(embed_dim, vocab_size, bias=False)
        
        # Precompute rotation matrix
        self.rotation_matr = self.decoder_blocks[0].attn.compute_rotation_matrix()
        
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
        x = self.word_embeddings(indices)  # shape (batch, seqlen, embed_dim)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, use_kv_cache, self.rotation_matr)

        x = self.rms_norm(x)
        logits = self.output_projection(x)
        
        return logits
    

    @staticmethod
    def from_pretrained(model_name: str):
        """Returns Mixtral with pretrained weights.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained model. Valid options are:
            * 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            * 'mistralai/Mixtral-8x7B-v0.1'

        """
        from .loadpretrained import load_pretrained_mixtral
        return load_pretrained_mixtral(model_name)
    
    
    def clear_kv_cache(self):
        """Clears kv_cache."""
        for decoder_block in self.decoder_blocks:
            decoder_block.attn.clear_kv_cache()
            
            
    def get_kv_cache_seqlen(self):
        """Gets KV cache seqlen."""
        return self.decoder_blocks[0].attn.get_kv_cache_seqlen()
                
    
class DecoderBlock(Module):
    
    def __init__(self,
                 embed_dim: int,
                 ffn_hidden_dim: int,
                 n_heads: int,
                 n_kv_heads: int,
                 n_experts: int,
                 n_experts_per_tok: int,
                 block_size: int,
                 rotary_base: int,
                 norm_eps: float):
        super().__init__()
        self.norm1 = candle.RMSNorm(axis=2, eps=norm_eps)
        self.attn = candle.GroupedQueryRotaryAttention(embed_dim, n_heads, n_kv_heads,
                                                       dropout_p=0.0,
                                                       apply_rotary_embedding=True,
                                                       rotary_base=rotary_base,
                                                       max_seqlen=block_size,
                                                       bias=False)
        self.norm2 = candle.RMSNorm(axis=2, eps=norm_eps)
        self.moe = MOE(embed_dim, ffn_hidden_dim, n_experts, n_experts_per_tok)

        
    def forward(self,
                x: Tensor,
                use_kv_cache: bool,
                rotation_matr: Tuple[Tensor, Tensor] = None):
        # x: Tensor with shape (batch, seqlen, embed_dim)
        x = x + self.self_attn(self.norm1(x), use_kv_cache, rotation_matr)
        x = x + self.moe(self.norm2(x))

        return x

    
    def self_attn(self,
                  x: Tensor,
                  use_kv_cache: bool,
                  rotation_matr: Tuple[Tensor, Tensor] = None):
        """Self-attention with causal mask."""
        # causal_attn_mask[i, j] = 0 means that query[i] attends to key[j], and so
        # causal_attn_mask[i, j] = 0 if i >= j and 1 otherwise.
        causal_attn_mask = Tensor(1 - np.tri(x.shape[1]))

        (attn_output, attn_scores) = self.attn(x, x, x, causal_attn_mask,
                                               use_kv_cache, rotation_matr)

        return attn_output


class MOE(Module):
    """Mixture of Experts layer."""
    
    def __init__(self,
                 input_dim: int,
                 ffn_hidden_dim: int,
                 n_experts: int,
                 n_experts_per_tok: int):
        super().__init__()

        self.experts = candle.ParameterList([
            FeedForwardBlock(input_dim, ffn_hidden_dim) for _ in range(n_experts)
        ])
        self.gate = candle.Linear(input_dim, n_experts, bias=False)
        self.n_experts_per_tok = n_experts_per_tok


    def forward(self, x):
        logits = self.gate(x)
        
        # weights, selected experts: (batch, seqlen, n_exp_per_tok)
        (weights, selected_experts) = F.topk(logits, self.n_experts_per_tok)
        weights = F.softmax(weights)
        
        # x_repeat: (batch, seqlen, n_exp_per_tok, input_dim)
        x_repeat = x.unsqueeze(2).repeat_interleave(self.n_experts_per_tok, axis=2)
        output = candle.empty_like(x_repeat)
        for (i, expert) in enumerate(self.experts):
            mask = (selected_experts == i)
            output[mask] = expert(x_repeat[mask]) * weights[mask].unsqueeze(1)
        
        output = output.sum(axis=2)  # Aggregate along `n_exp_per_tok` axis
        
        return output
    
    
class FeedForwardBlock(Module):
    
    def __init__(self,
                 input_dim: int,
                 ffn_hidden_dim: int):
        super().__init__()
        self.w1 = candle.Linear(input_dim, ffn_hidden_dim, bias=False)
        self.w2 = candle.Linear(ffn_hidden_dim, input_dim, bias=False)
        self.w3 = candle.Linear(input_dim, ffn_hidden_dim, bias=False)
        
        
    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w3(x)  # SwiGLU "activation"
        x = self.w2(x)
        
        return x
    
