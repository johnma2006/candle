import numpy as np
import pandas as pd
from typing import List, Tuple
from abc import ABC, abstractmethod

from .tensor import Tensor


class Parameter:
    
    def __init__(self,
                 weight: Tensor):
        self.weight = weight
        self.weight.requires_grad = True
        

    def __getattr__(self, attr):
        return getattr(self.weight, attr)
    
    
    def __setattr__(self, name, value):
        if name == 'weight':
            object.__setattr__(self, name, value)  # Call original __setattr__
        else:
            setattr(self.weight, name, value)
            

    def __repr__(self):
        return str(self.weight).replace('Tensor(', f'Parameter(')
    
    # ----------------------------
    # Redefine operation overloads
    # ----------------------------
    
    def __getitem__(self, key):
        return self.weight.__getitem__(key)
    
    
    def __add__(self, other):
        return self.weight.__add__(other)
    
    
    def __radd__(self, other):
        return self.weight.__radd__(other)
    
    
    def __sub__(self, other):
        return self.weight.__sub__(other)
    
    
    def __rsub__(self, other):
        return self.weight.__rsub__(other)
    

    def __mul__(self, other):
        return self.weight.__mul__(other)
    
    
    def __rmul__(self, other):
        return self.weight.__rmul__(other)
    
    
    def __truediv__(self, other):
        return self.weight.__truediv__(other)
    
    
    def __rtruediv__(self, other):
        return self.weight.__rtruediv__(other)
    
    
    def __neg__(self):
        return self.weight.__neg__()
    
    
    def __pos__(self):
        return self.weight.__pos__()
    
    
    def __pow__(self, power):
        return self.weight.__pow__(power)
    
    
    def __matmul__(self, other):
        return self.weight.__matmul__(other)
    
    
class HasParameters(ABC):
    """All classes that have Parameters as attributes should subclass this."""

    @abstractmethod
    def parameters():
        """Returns dictionary mapping parameter name to Parameter."""
        pass
    
    
class HasChildModules(ABC):
    """All classes that have Modules as attributes should subclass this."""

    @abstractmethod
    def child_modules():
        """Returns list of child modules."""
        pass
    
    
    def train(self,
              mode: bool = True):
        """Sets module and all child modules into training model."""
        self.training = mode
        for module in self.child_modules().values():
            module.train(mode)
            
            
    def eval(self):
        """Sets module and all child modules into eval model."""
        self.train(mode=False)
        
        
    def summary(self,
                input_shape: Tuple[int] = None,
                name_prefix: str = '',
                compute_output_shape: bool = False):
        """Returns DataFrame summarizing the model."""
        from .layers.module import Module
        
        if input_shape is not None:
            compute_output_shape = True
            
        if compute_output_shape and input_shape is not None:
            # Feed in fake input to initialize module._output_shape
            _ = self(Tensor(np.zeros(input_shape)))
        
        model_summary_df = pd.DataFrame(columns=['Layer Type', '# Parameters'])

        child_modules = self.child_modules()
        for attr_name in child_modules:
            module = child_modules[attr_name]

            if isinstance(module, Module):
                num_parameters = sum([np.prod(p.shape) for p in module.parameters().values()])
                model_summary_df.loc[f'{name_prefix}{attr_name}', 'Layer Type'] = type(module).__name__
                model_summary_df.loc[f'{name_prefix}{attr_name}', '# Parameters'] = num_parameters
                if compute_output_shape:
                    model_summary_df.loc[f'{name_prefix}{attr_name}', 'Output Shape'] = str(module._output_shape)
                    model_summary_df.loc[f'{name_prefix}{attr_name}', '_time'] = module._output_time

            elif isinstance(module, HasChildModules):
                model_summary_df = pd.concat([
                    model_summary_df,
                    module.summary(input_shape=None,
                                   name_prefix=f'{attr_name}.',
                                   compute_output_shape=compute_output_shape)
                ])

        if name_prefix == '':
            if compute_output_shape:
                model_summary_df = model_summary_df.sort_values('_time').drop('_time', axis=1)
                model_summary_df.loc['Total', 'Output Shape'] = ''
            model_summary_df.loc['Total', 'Layer Type'] = ''
            model_summary_df.loc['Total', '# Parameters'] = model_summary_df['# Parameters'].sum()

        return model_summary_df
    
    
class ParameterList(HasParameters, HasChildModules):
    
    def __init__(self, 
                 parameter_list: List[object]):
        """Initializes list of Parameters.
        
        Parameters
        ----------
        parameter_list
            List of Parameters or objects that subclass HasParameter.
            
        """
        for param in parameter_list:
            if not (isinstance(param, Parameter) or isinstance(param, HasParameters)):
                raise ValueError(f'Parameter {param} must be either class Parameter or HasParameter.')
            
        self.parameter_list = parameter_list
        
        
    def __getitem__(self, index):
        return self.parameter_list[index]
    
    
    def __len__(self):
        return len(self.parameter_list)
    
        
    def parameters(self):
        parameter_dict = {}
        for i in range(len(self.parameter_list)):
            param = self.parameter_list[i]
            if isinstance(param, Parameter):
                parameter_dict[str(i)] = param
                
            elif isinstance(param, HasParameters):
                attr_parameter_dict = param.parameters()
                for subattr_name in attr_parameter_dict:
                    parameter_dict[f'{str(i)}.{subattr_name}'] = attr_parameter_dict[subattr_name]
                    
        return parameter_dict
    
    
    def child_modules(self):
        """Returns dictionary of child modules."""
        from .layers.module import Module
        children = {}
        for (i, attr) in enumerate(self.parameter_list):
            if isinstance(attr, Module) or isinstance(attr, HasChildModules):
                children[str(i)] = attr
                    
        return children
    