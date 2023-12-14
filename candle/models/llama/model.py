"""Llama implementation.

References:
[1] Meta's LLaMA 2: https://github.com/facebookresearch/llama/blob/main/llama/

"""

import numpy as np
from typing import List, Tuple

import candle
import candle.functions as F
from candle.tensor import Tensor
from candle.layers.module import Module


class Llama(Module):
    
    def __init__(self,
                 n_layers: int,
                 n_heads: int,
                 embed_dim: int,
                 vocab_size: int,
                 block_size: int,
                 n_kv_heads: int = None,
                 ffn_hidden_dim: int = None,
                 rotary_base: int = 10000,
                 norm_eps: float = 1e-5):
        super().__init__()

        if n_kv_heads is None:
            n_kv_heads = n_heads
            
        if ffn_hidden_dim is None:
            # Conventionally, ffn_hidden_dim = 4 * embed_dim
            # Since LLaMA uses the SwiGLU "activation", ffn_hidden_dim is set to 4 * embed_dim * 2/3
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
                         block_size, rotary_base, norm_eps)
            for _ in range(n_layers)
        ])
        self.rms_norm = candle.RMSNorm(axis=2, eps=norm_eps)
        
        # Unlike GPT, LLaMA output layer is not tied to word embeddings
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
    def from_pretrained(model_name: str,
                        model_dir: str):
        """Returns LLaMA2 with pretrained weights.

        Parameters
        -----------
        model_name
            One of ['7b', '7b-chat', '13b', '13b-chat', '70b', '70b-chat'].
        model_dir
            Directory with LLaMA weights. To download, go to https://ai.meta.com/llama/.
            E.g. if model_dir = '/path/to/llama', the directory should look like this:

                /path/to/llama
                ├── tokenizer.model
                ├── tokenizer_checklist.chk
                ├── 7b
                │   ├── checklist.chk
                │   ├── consolidated.00.pth
                │   └── params.json
                ├── 7b-chat
                │   ├── checklist.chk
                │   ├── consolidated.00.pth
                │   └── params.json
                ├── 13b
                │   ├── checklist.chk
                │   ├── consolidated.00.pth
                │   ├── consolidated.01.pth
                │   └── params.json
                ├── 13b-chat
                │   ├─ ...
                │   ...

        Returns
        -------
        model
            LLaMA2 instance with Meta's weights initialized.

        """
        from .loadpretrained import load_pretrained_llama
        return load_pretrained_llama(model_name, model_dir)
    
    
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
        self.ffn = FeedForwardBlock(input_dim=embed_dim, ffn_hidden_dim=ffn_hidden_dim)

        
    def forward(self,
                x: Tensor,
                use_kv_cache: bool,
                rotation_matr: Tuple[Tensor, Tensor] = None):
        # x: Tensor with shape (batch, seqlen, embed_dim)
        x = x + self.self_attn(self.norm1(x), use_kv_cache, rotation_matr)
        x = x + self.ffn(self.norm2(x))

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
    