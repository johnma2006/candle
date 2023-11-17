import time
from abc import ABC, abstractmethod
from typing import Callable

from ..tensor import Tensor
from ..parameter import Parameter, HasParameters, HasChildModules


class Module(HasParameters, HasChildModules, ABC):
    
    def __init__(self):
        self.training = True
        self._hooks = []
        
        self._output_shape = None
        self._output_time = None
    
    
    @abstractmethod
    def forward(self):
        pass
    

    def parameters(self):
        """Returns dictionary mapping parameter name to Parameter."""
        parameter_dict = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                parameter_dict[attr_name] = attr
                
            elif isinstance(attr, HasParameters):
                attr_parameter_dict = attr.parameters()
                for subattr_name in attr_parameter_dict:
                    parameter_dict[f'{attr_name}.{subattr_name}'] = attr_parameter_dict[subattr_name]
                    
        return parameter_dict
    
    
    def child_modules(self):
        """Returns dictionary mapping name to child modules."""
        children = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module) or isinstance(attr, HasChildModules):
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
    
    
    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        
        # Apply hooks
        
        for hook_fn in self._hooks:
            _output = hook_fn(self, tuple(args), output)
            if _output is not None:
                output = _output
                
        # For model.summary()
        
        if isinstance(output, Tensor):
            self._output_shape = output.shape
        else:
            self._output_shape = None
        self._output_time = time.time_ns()
        
        return output
