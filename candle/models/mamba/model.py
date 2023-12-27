"""Mamba model. Adapted from my other repo at github.com/johnma2006/mamba-minimal.

References:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752

"""
from __future__ import annotations
import numpy as np
import math
import candle
import candle.functions as F
from candle import Tensor, Parameter, Module
from dataclasses import dataclass


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        
        self.embedding = candle.Embedding(args.vocab_size, args.d_model)
        self.layers = candle.ParameterList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = candle.RMSNorm(args.d_model)

        self.lm_head = self.embedding.embeddings  # Tie output projection to embedding weights.
                                                  # See "Weight Tying" paper


    def forward(self, input_ids, use_kv_cache: bool = False):
        """
        Args:
            input_ids (long tensor): shape (b, l)
            use_kv_cache (bool): dummy parameter to maintain compatibility with generation scripts.
    
        Returns:
            logits: shape (b, l, vocab_size)

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        logits = x @ self.lm_head.T

        return logits

    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from .loadpretrained import load_pretrained_mamba
        return load_pretrained_mamba(pretrained_model_name)
        

class ResidualBlock(Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = candle.RMSNorm(args.d_model)
        

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)
    
        Returns:
            output: shape (b, l, d)

        """
        output = self.mixer(self.norm(x)) + x

        return output
            

class MambaBlock(Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = candle.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d_weight = Parameter(candle.rand(args.d_inner, args.d_conv))
        self.conv1d_bias = Parameter(candle.rand(args.d_inner))

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = candle.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = candle.Linear(args.dt_rank, args.d_inner, bias=True)

        A = np.repeat(np.arange(1, args.d_state + 1)[None, :], args.d_inner, axis=0)
        self.A_log = Parameter(np.log(A))
        self.D = Parameter(np.ones(args.d_inner))
        self.out_proj = candle.Linear(args.d_inner, args.d_model, bias=args.bias)
        

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)
    
        Returns:
            output: shape (b, l, d)
        
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], axis=-1)
        
        x = self.conv1d(x)
        
        x = F.silu(x)
        
        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    
    def ssm(self, x):
        """Runs the SSM. See Algorithm 2 in Section 3.2 in the Mamba paper [1]

        Args:
            x: shape (b, l, d_in)
    
        Returns:
            output: shape (b, l, d_in)

        """
        (d_in, n) = self.A_log.shape
        
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -F.exp(self.A_log, base=np.e)
        D = self.D
        
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], axis=-1)
        
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        
        delta = delta.unsqueeze(3)  # (b l d_in 1)
        deltaA = F.exp(delta * A, base=np.e)
        deltaB_u = delta * B.unsqueeze(2) * u.unsqueeze(3)
        
        # Perform selective scan
        x = Tensor(np.zeros((b, d_in, n)))
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = F.bmm(x, C[:, i, :, None])
            ys.append(y)
        
        y = F.concat(ys, axis=2).transpose(1, 2)
        
        y = y + u * D

        return y
            
    
    def conv1d(self, x):
        """Poormans implementation of causal conv1d."""
        (b, l, d_in) = x.shape
        
        padding = Tensor(np.zeros((b, self.args.d_conv - 1, d_in)))
        x_padded = F.concat([padding, x], axis=1)
        
        x_conv = candle.zeros_like(x) + self.conv1d_bias
        for i in range(self.args.d_conv):
            x_scaled = x_padded * self.conv1d_weight[:, i]
            x_scaled = x_scaled[:, i:l + i]
            x_conv = x_conv + x_scaled

        return x_conv
