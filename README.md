Deep learning library, implemented from scratch in pure numpy for educational purposes.

#### Features:
* Tensor-based reverse-mode automatic differentiation
* Object-oriented PyTorch-like API
* [Tensor operations](https://github.com/johnma2006/candle/tree/main/candle/operations): slicing and reshaping, broadcasted arithmetic, tensor contractions, batch matmul
* [Layers](https://github.com/johnma2006/candle/tree/main/candle/layers): linear, multi-head attention, batch/layer/RMS(todo) norm, dropout, convolutional, max/avg pooling
* [NLP](https://github.com/johnma2006/candle/tree/main/candle/nlp): byte-pair encoding, beam search, speculative sampling (todo), nucleus sampling
* Models: [GPT](https://github.com/johnma2006/candle/blob/main/candle/models/gpt/model.py), [ResNet](https://github.com/johnma2006/candle/blob/main/candle/models/resnet/model.py)
* Optimizers: SGD, AdamW
* LR schedulers: step decay, cosine annealing, warmup
* Lightweight Tensorboard-like dashboarding
* Focus on readable, understandable, idiomatic code


## Demos & Experiments

#### Language Modelling
* Converse with Taylor, your Large GPT2 friend [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/gpt_experiments/1.0%20Converse%20with%20Taylor%2C%20your%20Large%20GPT2%20friend.ipynb)
 ![Sample conversation with Taylor](https://github.com/johnma2006/candle/blob/main/experiments/gpt_experiments/sample_conversation.png)

#### Vision
* Training a ResNet14 on MNIST [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/vision_experiments/2.0%20ResNet14%20on%20MNIST.ipynb)
* Training an MLP on MNIST [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/vision_experiments/1.0%20MLP%20on%20MNIST%20-%20AdamW.ipynb)

#### Initialization
* Gradient Norm vs. Model {Depth, Norm} under {Xavier, Kaiming} init
  * Width, Kaiming  [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Width%20on%20Gradient%20Norm%20-%20MLP%20with%20Kaiming%20Init.ipynb)
  * Width, Xavier  [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Width%20on%20Gradient%20Norm%20-%20MLP%20with%20Kaiming%20Init.ipynb)
  * Depth, Kaiming [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Depth%20on%20Gradient%20Norm%20-%20MLP%20with%20Xavier%20Init.ipynb)
  * Depth, Xavier [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/2.0%20Effect%20of%20Model%20Depth%20on%20Gradient%20Norm%20-%20MLP%20with%20Xavier%20Init.ipynb)
* Activation Distributions vs Init [(notebook)](https://github.com/johnma2006/candle/blob/main/experiments/initialization_experiments/1.0%20Activation%20Distribution%20by%20Layer%20w.r.t%20Initialization.ipynb)


## Example GPT2 Implementation

```python
import numpy as np
import candle
import candle.functions as F
from candle import Module, Tensor


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
        self.embed_dim = embed_dim
        self.block_size = block_size
        
        self.dropout = candle.Dropout(dropout_p)
        self.word_embeddings = candle.Embedding(vocab_size, embed_dim)
        self.position_embeddings = candle.Embedding(block_size, embed_dim)
        self.decoder_blocks = candle.ParameterList([DecoderBlock(embed_dim, num_heads, dropout_p)
                                                    for _ in range(num_layers)])
        self.layer_norm = candle.LayerNorm(axis=2)
        
        # Tie output projection weights to word embeddings. See "Weight Tying" paper.
        self.output_projection = self.word_embeddings.embeddings
        
    
    def forward(self, indices):
        position_indices = Tensor(np.arange(indices.shape[1]))
        
        x = self.word_embeddings(indices) + self.position_embeddings(position_indices)
        x = self.dropout(x)  # x: shape (batch, seqlen, embed_dim)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        x = self.layer_norm(x)
        
        return x @ self.output_projection.T

    
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
```
```python
model = GPT(num_layers=12,
            num_heads=12,
            embed_dim=768,
            vocab_size=50257,
            block_size=1024,
            dropout_p=0.1)

tokenizer = candle.models.gpt.GPT2BPETokenizer()
indices = candle.Tensor([tokenizer.encode(
    'Once upon a time, there is a cat whose name is Maukoo. He loves eating and cuddling.'
)])

# Example backpropagation

targets = indices[:, 1:]
logits = model(indices[:, :-1])
loss = F.cross_entropy_loss(logits, targets)
loss.backward()

# Example generation

model = candle.models.gpt.GPT.from_pretrained('gpt2-large')

generator = candle.nlp.beam_search_decoder(model, indices[0],
                                           n_tokens_to_generate=50,
                                           beam_size=1,
                                           top_p=0.90,  # Nucleus sampling
                                           top_k=100)

response_indices = np.concatenate(list(generator))

print(tokenizer.decode(response_indices))
# Output:  A lot.  He also loves drinking.  (But it's an odd habit for a cat that loves eating
# and cuddling.)  This little kitty is not the sort of kitty you would expect to be a
```


## Run Tests

`python -m unittest`
