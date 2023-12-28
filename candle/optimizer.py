import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Optimizer(ABC):
    
    def __init__(self,
                 parameter_dict: dict):
        self.parameter_dict = parameter_dict
        self.scheduler = None
    
    
    @abstractmethod
    def step(self):
        pass
    
    
    def zero_grad(self, set_to_none: bool = True):
        """Resets grad to None."""
        for param in self.parameter_dict.values():
            if set_to_none:
                param.grad = None
            else:
                param.grad = 0.0
                
    
    def get_learning_rate(self):
        if self.scheduler is None:
            return self.learning_rate
        else:
            return self.scheduler.get_learning_rate()
            
            
    def get_initial_learning_rate(self):
        return self.learning_rate
            
    
    def set_scheduler(self, scheduler):
        if self.scheduler is None:
            self.scheduler = scheduler
        else:
            raise ValueError(f'Scheduler already set for this optimizer.')
    
    
class SGD(Optimizer):
    """SGD with momentum."""
    
    def __init__(self,
                 parameter_dict: dict,
                 learning_rate: float,
                 momentum: float = 0.0,
                 weight_decay: float = 0.0):
        super().__init__(parameter_dict)
        self.learning_rate = learning_rate
        self.mom = momentum
        self.weight_decay = weight_decay
        
        self.momentum = {key: None for key in self.parameter_dict}
    
    
    def step(self):
        for parameter_name in self.parameter_dict:
            param = self.parameter_dict[parameter_name]
            
            if param.grad is None:
                continue  # Skip step, see https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

            if self.momentum[parameter_name] is None:
                self.momentum[parameter_name] = param.grad
            else:
                self.momentum[parameter_name] = (self.mom * self.momentum[parameter_name]
                                                 + (1 - self.mom) * param.grad)

            param.data -= (self.get_learning_rate() * self.momentum[parameter_name]
                           + self.get_learning_rate() * self.weight_decay * param.data)
            
            
class AdamW(Optimizer):
    """Adam with decoupled weight decay.
    
    References:
        [1] Ilya Loshchilov, Frank Hutter.
            Decoupled Weight Decay Regularization. arXiv:1711.05101, 2017.

    """
    
    def __init__(self,
                 parameter_dict: dict,
                 learning_rate: float,
                 weight_decay: float = 0.0,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8):
        super().__init__(parameter_dict)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.betas = betas
        self.eps = eps
        
        self.momentum = {key: 0.0 for key in self.parameter_dict}
        self.variance = {key: 0.0 for key in self.parameter_dict}
        self.t = 0
        
    
    def step(self):
        self.t += 1

        for parameter_name in self.parameter_dict:
            param = self.parameter_dict[parameter_name]
            
            if param.grad is None:
                continue  # Skip step, see https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            
            # Update momentum and variance

            self.momentum[parameter_name] = (self.betas[0] * self.momentum[parameter_name]
                                             + (1 - self.betas[0]) * param.grad)

            self.variance[parameter_name] = (self.betas[1] * self.variance[parameter_name]
                                             + (1 - self.betas[1]) * param.grad ** 2)

            # Update parameters
            
            update = self.compute_update(self.momentum[parameter_name], self.variance[parameter_name], self.t)
            param.data -= self.get_learning_rate() * update + self.get_learning_rate() * self.weight_decay * param.data
            
            
    def compute_update(self, momentum, variance, t):
        m_unbiased = momentum / (1 - self.betas[0] ** t)
        v_unbiased = variance / (1 - self.betas[1] ** t)
        update = m_unbiased / (np.sqrt(v_unbiased) + self.eps)

        return update
    