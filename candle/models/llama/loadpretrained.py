"""Functionality to load LLaMA2 with Meta's LLaMA-2 weights"""

import json
from pathlib import Path


LLAMA_CONFIG_BY_SIZE = {
    '7b': dict(n_layers=32, n_heads=32, n_kv_heads=None, embed_dim=4096, vocab_size=32000, block_size=2048),
    '13b': dict(n_layers=40, n_heads=40, n_kv_heads=None, embed_dim=5120, vocab_size=32000, block_size=2048),
    '70b': dict(n_layers=80, n_heads=64, n_kv_heads=8, embed_dim=8192, vocab_size=32000, block_size=2048, ffn_hidden_dim=28672),
}


def load_pretrained_llama(model_name: str,
                          model_dir: str):
    """Loads a pre-trained LLaMA2 model with Meta's weights.

    Parameters
    ----------
    model_name : str
        Name of the pre-trained LLaMA2 model. Valid options are:
        * '7b'
        * '7b-chat'
        * '13b'
        * '13b-chat'
        * '70b'
        * '70b-chat'

    model_dir : str
        Directory containing the pre-trained LLaMA2 model weights.
        The directory structure should be as follows:

        /path/to/llama
        ├── tokenizer.model
        ├── tokenizer_checklist.chk
        ├── llama-2-7b
        │   ├── checklist.chk
        │   ├── consolidated.00.pth
        │   └── params.json
        ├── llama-2-7b-chat
        │   ├── checklist.chk
        │   ├── consolidated.00.pth
        │   └── params.json
        ├── llama-2-13b
        │   ├── checklist.chk
        │   ├── consolidated.00.pth
        │   ├── consolidated.01.pth
        │   └── params.json
        ├── llama-2-13b-chat
        │   ...
        ...

    Returns
    -------
    model : Llama
        A Llama model instance with Meta's pre-trained weights loaded.

    """
    from .model import Llama
    import torch
    assert model_name in ['7b', '7b-chat', '13b', '13b-chat', '70b', '70b-chat']
    model_dir = Path(model_dir)

    # -------------------
    # Load Meta's weights
    # -------------------

    state_dict_paths = sorted([i for i in (model_dir / f'llama-2-{model_name}').iterdir() if 'consolidated' in str(i)])

    state_dict = {}
    for state_dict_path in state_dict_paths:
        state_dict_shard = torch.load(str(state_dict_path), map_location='cpu', mmap=True, weights_only=True)

        # Weight for one param may be in multiple shards for some reason; concat them
        for param_name in state_dict_shard:
            if param_name not in state_dict:
                state_dict[param_name] = state_dict_shard[param_name]
            elif not param_name.endswith('norm.weight'):
                if (param_name == 'tok_embeddings.weight'
                    or param_name.endswith('attention.wo.weight')
                    or param_name.endswith('feed_forward.w2.weight')):
                    dim = 1
                else:
                    dim = 0
                state_dict[param_name] = torch.cat([state_dict[param_name],
                                                    state_dict_shard[param_name]],
                                                   dim=dim)
        del state_dict_shard

    # -----------------------------
    # Initialize candle.Llama model
    # -----------------------------

    with open(model_dir / f'llama-2-{model_name}' / 'params.json') as f:
        model_config = json.load(f)

    model_size = model_name.split('-')[0]
    model = Llama(**LLAMA_CONFIG_BY_SIZE[model_size], norm_eps=model_config['norm_eps'])
    _ = model.summary((1, 1))  # To initialize lazy params

    # ----------------
    # Transfer weights
    # ----------------

    def meta_param(name):
        return state_dict.pop(name).float().numpy().copy()

    params = dict(model.parameters())

    params['word_embeddings.embeddings'].data[:] = meta_param('tok_embeddings.weight')
    params['rms_norm.W'].data[:] = meta_param('norm.weight')
    params['output_projection.W'].data[:] = meta_param('output.weight').T

    for i in range(model.n_layers):
        params[f'decoder_blocks.{i}.attn.W_q.W'].data[:] = meta_param(f'layers.{i}.attention.wq.weight').T
        params[f'decoder_blocks.{i}.attn.W_k.W'].data[:] = meta_param(f'layers.{i}.attention.wk.weight').T
        params[f'decoder_blocks.{i}.attn.W_v.W'].data[:] = meta_param(f'layers.{i}.attention.wv.weight').T
        params[f'decoder_blocks.{i}.attn.W_o.W'].data[:] = meta_param(f'layers.{i}.attention.wo.weight').T

        params[f'decoder_blocks.{i}.ffn.w1.W'].data[:] = meta_param(f'layers.{i}.feed_forward.w1.weight').T
        params[f'decoder_blocks.{i}.ffn.w2.W'].data[:] = meta_param(f'layers.{i}.feed_forward.w2.weight').T
        params[f'decoder_blocks.{i}.ffn.w3.W'].data[:] = meta_param(f'layers.{i}.feed_forward.w3.weight').T

        params[f'decoder_blocks.{i}.norm1.W'].data[:] = meta_param(f'layers.{i}.attention_norm.weight')
        params[f'decoder_blocks.{i}.norm2.W'].data[:] = meta_param(f'layers.{i}.ffn_norm.weight')
        
    return model
                          