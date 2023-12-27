import torch
import json
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_pretrained_mamba(pretrained_model_name: str):
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
    from .model import Mamba, ModelArgs
    
    def load_config_hf(model_name):
        resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return json.load(open(resolved_archive_file))
    
    
    def load_state_dict_hf(model_name, device=None, dtype=None):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
    
    config_data = load_config_hf(pretrained_model_name)
    args = ModelArgs(
        d_model=config_data['d_model'],
        n_layer=config_data['n_layer'],
        vocab_size=config_data['vocab_size']
    )
    model = Mamba(args)
    
    state_dict = load_state_dict_hf(pretrained_model_name)
    
    def mamba_param(name):
        return state_dict.pop(name).float().numpy().copy()
    
    params = dict(model.parameters())
    
    params['embedding.embeddings'].data[:] = mamba_param('backbone.embedding.weight')
    params['norm_f.W'].data[:] = mamba_param('backbone.norm_f.weight')
    
    for layer_i in range(args.n_layer):
        params[f'layers.{layer_i}.mixer.D'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.D')
        params[f'layers.{layer_i}.mixer.in_proj.W'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.in_proj.weight').T
        params[f'layers.{layer_i}.mixer.conv1d_weight'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.conv1d.weight')[:, 0]
        params[f'layers.{layer_i}.mixer.conv1d_bias'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.conv1d.bias')
        params[f'layers.{layer_i}.mixer.x_proj.W'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.x_proj.weight').T
        params[f'layers.{layer_i}.mixer.dt_proj.W'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.dt_proj.weight').T
        params[f'layers.{layer_i}.mixer.dt_proj.b'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.dt_proj.bias')
        params[f'layers.{layer_i}.mixer.A_log'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.A_log')
        params[f'layers.{layer_i}.mixer.out_proj.W'].data[:] = mamba_param(f'backbone.layers.{layer_i}.mixer.out_proj.weight').T
        params[f'layers.{layer_i}.norm.W'].data[:] = mamba_param(f'backbone.layers.{layer_i}.norm.weight')
    
    return model
    