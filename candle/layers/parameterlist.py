from typing import List
from .module import Module
from ..tensor import Parameter


class ParameterList(Module):
    
    def __init__(self, 
                 parameter_list: List[object]):
        """Initializes list of Parameters.
        
        Args:
            parameter_list (List[Parameter or Module]): List of Parameters or Modules:
            
        """
        super().__init__()
        for param in parameter_list:
            if not isinstance(param, Parameter) and not isinstance(param, Module):
                raise ValueError(f'{param} must be either class Parameter or Module.')
            
        self.parameter_list = parameter_list
        self._extra_levels_to_expand = 1
        
        
    def forward(self):
        raise RuntimeError('ParameterList cannot be called.')
        
        
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
                
            elif isinstance(param, Module):
                attr_parameter_dict = param.parameters()
                for subattr_name in attr_parameter_dict:
                    parameter_dict[f'{str(i)}.{subattr_name}'] = attr_parameter_dict[subattr_name]
                    
        # Deduplicate parameter dict in case of weight tying
        parameter_dict = self.deduplicate_parameter_dict(parameter_dict)
                    
        return parameter_dict
    
    
    def child_modules(self):
        """Returns dictionary of child modules."""
        children = {}
        for (i, attr) in enumerate(self.parameter_list):
            if isinstance(attr, Module):
                children[str(i)] = attr
                    
        return children
    