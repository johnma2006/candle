import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Callable, Tuple

from ..tensor import Tensor, Parameter


class Module(ABC):
    
    def __init__(self):
        """If the subclass overrides __init__(), it must make sure to invoke super().__init__()."""
        self.training = True
        self._hooks = []
        self._output_shape = None
        self._extra_levels_to_expand = 0  # How many extra levels to expand in model.summary
    
    
    @abstractmethod
    def forward(self):
        pass
    
    
    def zero_grad(self, set_to_none: bool = True):
        """Resets grad to None."""
        for param in self.parameters().values():
            if set_to_none:
                param.grad = None
            else:
                param.grad = 0.0
                

    def parameters(self):
        """Returns dictionary mapping parameter name to Parameter."""
        parameter_dict = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                parameter_dict[attr_name] = attr
                
            elif isinstance(attr, Module):
                attr_parameter_dict = attr.parameters()
                for subattr_name in attr_parameter_dict:
                    parameter_dict[f'{attr_name}.{subattr_name}'] = attr_parameter_dict[subattr_name]
                    
        # Deduplicate parameter dict in case of weight tying
        parameter_dict = self.deduplicate_parameter_dict(parameter_dict)
                    
        return parameter_dict
    
    
    def child_modules(self):
        """Returns dictionary mapping name to direct child modules."""
        children = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                children[attr_name] = attr
                    
        return children
    
    
    def register_forward_hook(self, hook_fn):
        """Registers a hook that will be called every time forward() is completed.

        Parameters
        ----------
        hook_fn
            Callable with the following signature:
                def hook_fn(module, input, output) -> None or modified output

        """
        if not callable(hook_fn):
            raise ValueError('hook_fn must be Callable function with signature `def hook_fn(module, input, output)`.')
            
        self._hooks.append(hook_fn)
    
    
    def train(self,
              mode: bool = True):
        """Sets module and all child modules into training model."""
        self.training = mode
        for module in self.child_modules().values():
            module.train(mode)
            
            
    def eval(self):
        """Sets module and all child modules into eval model."""
        self.train(mode=False)
    
    
    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        
        # Apply hooks
        
        if not hasattr(self, '_hooks'):
            raise Exception(f'Module {type(self).__name__} must call super().__init__() if overriding the constructor.')
        
        for hook_fn in self._hooks:
            _output = hook_fn(self, tuple(args), output)
            if _output is not None:
                output = _output
        
        # For model.summary()
        
        if isinstance(output, Tensor):
            self._output_shape = output.shape
        else:
            self._output_shape = None
        
        return output
    

    def deduplicate_parameter_dict(self, parameter_dict):
        """When weights are tied together, we need to deduplicate the parameter dict.

        We do that by choosing the parameter that comes first, alphabetically.

        Example:
            self.param1 = Parameter(Tensor([1, 2, 3]))
            self.param2 = param1

            We want model.parameters() to only return param1, not both param1 and param2.

        """
        dedup_parameter_dict = {}
        seen = set()
        for param_name in sorted(parameter_dict):
            param = parameter_dict[param_name]

            if id(param) not in seen:
                dedup_parameter_dict[param_name] = param
                seen.add(id(param))

        return dedup_parameter_dict
        
        
    def summary(self,
                input_shape: Tuple[int] = None,
                expand_submodules_to_level: int = 0,
                _level: int = 0):
        """Returns DataFrame summarizing the model.
        
        Parameters
        ----------
        input_shape
            If provided, then the output shape of each module will be computed.
        expand_submodules_to_level
            How far recursively into each module to expand. None to expand everything.
            
        """
        if expand_submodules_to_level is None:
            expand_submodules_to_level = np.inf
            
        if input_shape is not None and _level == 0:
            # Feed in fake input to initialize module._output_shape
            training_mode = self.training
            self.eval()  # Set to eval mode to prevent model from changing from e.g. batch norm ema_mean update
            _ = self(Tensor(np.zeros(input_shape)))
            self.train(mode=training_mode)

        columns = ['Layer Type', '# Parameters']
        if input_shape is not None:
            columns += ['Output Shape']
            
        model_summary_df = pd.DataFrame(columns=columns)

        # Add child parameters to summary

        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                num_parameters = np.prod(attr.shape)
                model_summary_df.loc[f'{attr_name}', 'Layer Type'] = 'Parameter'
                model_summary_df.loc[f'{attr_name}', '# Parameters'] = num_parameters
                if input_shape is not None:
                    model_summary_df.loc[attr_name, 'Output Shape'] = None

        # Add child modules to summary

        child_modules = self.child_modules()
        for attr_name in child_modules:
            module = child_modules[attr_name]

            submodule_summary_df = module.summary(input_shape=input_shape,
                                                  expand_submodules_to_level=expand_submodules_to_level,
                                                  _level=_level + 1)

            if _level < expand_submodules_to_level + module._extra_levels_to_expand and not submodule_summary_df.empty:
                submodule_summary_df.index = submodule_summary_df.index.map(lambda x: f'{attr_name}.{x}')
            else:
                # Condense submodule_summary_df into one line
                num_parameters = submodule_summary_df['# Parameters'].sum()
                submodule_summary_df = pd.DataFrame(columns=columns)
                submodule_summary_df.loc[attr_name, 'Layer Type'] = type(module).__name__
                submodule_summary_df.loc[attr_name,  '# Parameters'] = num_parameters
                if input_shape is not None:
                    submodule_summary_df.loc[attr_name, 'Output Shape'] = str(module._output_shape)

            model_summary_df = pd.concat([model_summary_df, submodule_summary_df])

        # If at top level, split index into multiple columns and add a "Total" summary row
            
        if _level == 0 and len(model_summary_df) > 0:
            multi_index = [i.split('.') for i in model_summary_df.index]
            max_level = max([len(i) for i in multi_index])
            multi_index = [i + [''] * (max_level - len(i)) for i in multi_index]
            multi_index = pd.DataFrame(multi_index, index=model_summary_df.index)

            model_summary_df = pd.concat([multi_index, model_summary_df], axis=1)
            model_summary_df = model_summary_df.sort_values(list(multi_index.columns)).set_index(list(multi_index.columns))
            model_summary_df.index.names = [''] * len(model_summary_df.index.names)
            
            if input_shape is not None:
                model_summary_df.loc['Total', 'Output Shape'] = ''
            model_summary_df.loc['Total', 'Layer Type'] = ''
            model_summary_df.loc['Total', '# Parameters'] = sum([np.prod(p.shape) for p in self.parameters().values()])

        return model_summary_df
    