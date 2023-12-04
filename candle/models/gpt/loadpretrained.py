"""Functionality to load GPT2 with OpenAI's pre-trained weights"""

import numpy as np
import transformers

from candle.tensor import Tensor


def load_pretrained_gpt(model_name: str):
    """Returns GPT2 with pretrained weights.

    Parameters
    -----------
    model_name
        One of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'].
            gpt2:         124M params
            gpt2-medium:  354M params
            gpt2-large:   774M params
            gpt2-xl:    1,557M params

    Returns
    -------
    model
        GPT instance with pre-trained weights initialized.

    """
    from .model import GPT
    
    model_names = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    if model_name not in model_names:
        raise ValueError(f'Invalid model name \'{model_names}\', must be one of {model_names}.')

    model_hf = transformers.GPT2LMHeadModel.from_pretrained(model_name)

    config = {
        'n_layers': len(model_hf.transformer.h),
        'n_heads': model_hf.transformer.h[0].attn.num_heads,
        'embed_dim': model_hf.transformer.h[0].attn.embed_dim,
        'vocab_size': model_hf.lm_head.out_features,
        'block_size': model_hf.transformer.wpe.num_embeddings,
        'dropout_p': model_hf.transformer.drop.p
    }

    model = GPT(**config)
    _ = model(Tensor(np.zeros((1, 32))))  # Feed fake batch to initialized deferred params

    # -----------------------
    # Transfer OpenAI weights
    # -----------------------

    def openai_param(name):
        openai_parameters = dict(model_hf.named_parameters())
        return openai_parameters[f'transformer.{name}'].detach().numpy()

    params = model.parameters()

    # Transfer embedding weights

    params['output_projection'].data[:] = openai_param('wte.weight')  # tied to word_embeddings.embeddings
    params['position_embeddings.embeddings'].data[:] = openai_param('wpe.weight')

    # Transfer decoder weights

    for i in range(len(model.decoder_blocks)):
        attn_weight = openai_param(f'h.{i}.attn.c_attn.weight').reshape((-1, 3, config['embed_dim']))
        attn_bias = openai_param(f'h.{i}.attn.c_attn.bias').reshape((3, config['embed_dim']))
        for (j, qkv) in enumerate(['q', 'k', 'v']):
            params[f'decoder_blocks.{i}.attn.W_{qkv}.W'].data[:] = attn_weight[:, j]
            params[f'decoder_blocks.{i}.attn.W_{qkv}.b'].data[:] = attn_bias[j]
            
        params[f'decoder_blocks.{i}.attn.W_o.W'].data[:] = openai_param(f'h.{i}.attn.c_proj.weight')
        params[f'decoder_blocks.{i}.attn.W_o.b'].data[:] = openai_param(f'h.{i}.attn.c_proj.bias')

        params[f'decoder_blocks.{i}.ffn.linear1.W'].data[:] = openai_param(f'h.{i}.mlp.c_fc.weight')
        params[f'decoder_blocks.{i}.ffn.linear1.b'].data[:] = openai_param(f'h.{i}.mlp.c_fc.bias')
        params[f'decoder_blocks.{i}.ffn.linear2.W'].data[:] = openai_param(f'h.{i}.mlp.c_proj.weight')
        params[f'decoder_blocks.{i}.ffn.linear2.b'].data[:] = openai_param(f'h.{i}.mlp.c_proj.bias')

        params[f'decoder_blocks.{i}.ln1.W'].data[0, 0, :] = openai_param(f'h.{i}.ln_1.weight')
        params[f'decoder_blocks.{i}.ln1.b'].data[0, 0, :] = openai_param(f'h.{i}.ln_1.bias')
        params[f'decoder_blocks.{i}.ln2.W'].data[0, 0, :] = openai_param(f'h.{i}.ln_2.weight')
        params[f'decoder_blocks.{i}.ln2.b'].data[0, 0, :] = openai_param(f'h.{i}.ln_2.bias')

    # Transfer final layer norm weights

    params['layer_norm.W'].data[0, 0, :] = openai_param('ln_f.weight')
    params['layer_norm.b'].data[0, 0, :] = openai_param('ln_f.bias')

    return model
