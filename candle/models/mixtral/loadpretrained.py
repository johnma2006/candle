import candle


def load_pretrained_mixtral(model_name: str):
    """Loads a pre-trained Mixtral model with Mistral's pre-trained weights.
    
    Note: currently the implementation is SUPER memory inefficient requiring peak 2x the model param size.
    Todo is to implement load_state_dicts to optimize speed/memory.

    Args:
        model_name (str): Name of the pre-trained model. Valid options are:
            * 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            * 'mistralai/Mixtral-8x7B-v0.1'
        
    Returns:
        Mixtral model instance with Mistral's pre-trained weights loaded.

    """
    from .model import Mixtral
    import transformers
    
    model_names = ['mistralai/Mixtral-8x7B-Instruct-v0.1', 'mistralai/Mixtral-8x7B-v0.1']
    if model_name not in model_names:
        raise ValueError(f'Invalid model name \'{model_names}\' must be one of {model_names}.')
    
    model_hf = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    
    config = {
        'n_layers': len(model_hf.model.layers),
        'n_heads': model_hf.model.layers[0].self_attn.num_heads,
        'embed_dim': model_hf.model.layers[0].hidden_size,
        'n_experts': model_hf.num_experts,
        'n_experts_per_tok': model_hf.num_experts_per_tok,
        'vocab_size': model_hf.vocab_size,
        'block_size': model_hf.model.layers[0].self_attn.max_position_embeddings,
        'n_kv_heads': model_hf.model.layers[0].self_attn.num_key_value_heads,
        'ffn_hidden_dim': model_hf.model.layers[0].block_sparse_moe.ffn_dim,
        'rotary_base': model_hf.model.layers[0].self_attn.rope_theta,
        'norm_eps': model_hf.model.norm.variance_epsilon,
    }
    
    model = Mixtral(**config)
    params = model.parameters()
    
    # ------------------------
    # Transfer Mistral weights
    # ------------------------
    
    mistral_params = dict(model_hf.named_parameters())
    del model_hf
    
    def mistral_param(name):
        param = mistral_params.pop(name).detach().numpy()
        if param.dtype != candle.Tensor.DEFAULT_DTYPE:
            param = param.astype(candle.Tensor.DEFAULT_DTYPE)
        return param
    
    def permute_query_key_projs(w, n_heads, kv_dim, embed_dim):
        """HuggingFace implemention of rotary attention uses the bisection impl instead of the interleaved impl.
        As a result, we need to do some permutations of the query/key projections to get things back to 'normal'."""
        per_head_dim = kv_dim // n_heads
        return w.reshape(n_heads, 2, per_head_dim // 2, embed_dim).swapaxes(1, 2).reshape(kv_dim, embed_dim)
    
    kv_dim = config['embed_dim'] // config['n_heads'] * config['n_kv_heads']
    
    params['word_embeddings.embeddings'].data[:] = mistral_param('model.embed_tokens.weight')
    params['rms_norm.W'].data[:] = mistral_param('model.norm.weight')
    params['output_projection.W'].data[:] = mistral_param('lm_head.weight').T
    
    for layer_i in range(config['n_layers']):    
        params[f'decoder_blocks.{layer_i}.attn.W_q.W'].data[:] = permute_query_key_projs(
            mistral_param(f'model.layers.{layer_i}.self_attn.q_proj.weight'),
            config['n_heads'], config['embed_dim'], config['embed_dim']
        ).T
        params[f'decoder_blocks.{layer_i}.attn.W_k.W'].data[:] = permute_query_key_projs(
            mistral_param(f'model.layers.{layer_i}.self_attn.k_proj.weight'),
            config['n_kv_heads'], kv_dim, config['embed_dim']
        ).T
        params[f'decoder_blocks.{layer_i}.attn.W_v.W'].data[:] = mistral_param(f'model.layers.{layer_i}.self_attn.v_proj.weight').T
        params[f'decoder_blocks.{layer_i}.attn.W_o.W'].data[:] = mistral_param(f'model.layers.{layer_i}.self_attn.o_proj.weight').T
        
        params[f'decoder_blocks.{layer_i}.moe.gate.W'].data[:] = mistral_param(f'model.layers.{layer_i}.block_sparse_moe.gate.weight').T
        for expert_i in range(config['n_experts']):
            params[f'decoder_blocks.{layer_i}.moe.experts.{expert_i}.w1.W'].data[:] = mistral_param(f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w1.weight').T
            params[f'decoder_blocks.{layer_i}.moe.experts.{expert_i}.w2.W'].data[:] = mistral_param(f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w2.weight').T
            params[f'decoder_blocks.{layer_i}.moe.experts.{expert_i}.w3.W'].data[:] = mistral_param(f'model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.w3.weight').T    
        
        params[f'decoder_blocks.{layer_i}.norm1.W'].data[:] = mistral_param(f'model.layers.{layer_i}.input_layernorm.weight')
        params[f'decoder_blocks.{layer_i}.norm2.W'].data[:] = mistral_param(f'model.layers.{layer_i}.post_attention_layernorm.weight')
    
    return model
    