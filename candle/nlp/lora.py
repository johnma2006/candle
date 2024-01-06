"""Simple implementation of the LoRA adapter.

References:
[1] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen.
    LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021
    
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .. import Module, Parameter, Dropout, Linear, weightinit


@dataclass
class LoraConfig:
    rank: int
    alpha: float
    dropout: float = 0.0
    keys: List[str] = None  # If not None, then only wraps linear layers if the name of
                            # the layer contains any of the strings in `keys`.
    

class LoraLinear(Module):
    
    def __init__(self, base_linear, config):
        super().__init__()
        self.base_linear = base_linear
        self.config = config
        self.scaling = self.config.alpha / self.config.rank

        self.A = Parameter(weightinit.kaiming((base_linear.input_nodes, config.rank)))
        self.B = Parameter(np.zeros((config.rank, base_linear.output_nodes)))
        self.dropout = Dropout(config.dropout)

    
    def forward(self, x):
        x_lora = (self.dropout(x) @ self.A @ self.B) * self.scaling

        return self.base_linear(x) + x_lora


def lora_wrapper(base_model, config: LoraConfig):
    """Modifies the base model in-place with linear layers replaced by LoRA linear layers.

    Args:
        base_model (Module). e.g. GPT or Llama.
        config (LoraConfig). Lora configs.

    Returns:
        Model. base_model, modified in-place. 
        
    """
    # Freeze all pre-trained model parameters
    for param in base_model.parameters().values():
        param.requires_grad = False
    
    # Replace linear layers by LoRA linear layers
    module_queue = [(None, base_model)]
    while len(module_queue) > 0:
        (module_name, module) = module_queue.pop()
        child_modules = module.child_modules()
        
        for name in child_modules:
            child_name = f'{module_name}.{name}' if module_name is not None else name
            if isinstance(child_modules[name], Linear):
                if config.keys is None or np.any([key in child_name for key in config.keys]):
                    lora_linear = LoraLinear(base_linear=getattr(module, name), config=config)
                    setattr(module, name, lora_linear)
            elif isinstance(child_modules[name], Module):
                module_queue.append((child_name, child_modules[name]))

    return base_model


def load_lora_adapter(lora_model, lora_adapter: Dict[str, Parameter]):
    """Loads LoRA adapter into LoRA model.
    
    Args:
        lora_model (Module). The result of lora_model = lora_wrapper(base_model, config).
        lora_adapter (Dict[str, Parameter]). The result of calling lora_model.parameters().

    Returns:
        Module. lora_model with lora_adapter loaded in.
        
    """
    params = lora_model.parameters()
    for name in params:
        params[name].data[:] = lora_adapter[name].data

    return lora_model
