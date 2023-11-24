"""Learning rate schedulers."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List


class Scheduler(ABC):
    
    def __init__(self, optimizer=None):
        self.T = 0
        if optimizer is not None:
            self.optimizer = optimizer
            optimizer.set_scheduler(self)
        else:
            self.optimizer = None
    
    
    @abstractmethod
    def get_learning_rate_at_T(self, T: int):
        """Get learning rate at step T."""
        pass
    
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.get_learning_rate_at_T(self.T)
    
    
    def step(self):
        """Increment internel step count."""
        self.T += 1


class StepLR(Scheduler):
    """Decay the learning rate by `gamma` every `step_size` iterations."""
    
    def __init__(self,
                 optimizer,
                 step_size: int,
                 gamma: float):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        
        self.max_learning_rate = optimizer.get_initial_learning_rate()
    
    
    def get_learning_rate_at_T(self, T: int):
        return self.max_learning_rate * self.gamma ** (T // self.step_size)
    
    
class MultiStepLR(Scheduler):
    """Decay the learning rate by `gamma` every time a milestone is passed."""
    
    def __init__(self,
                 optimizer,
                 milestones: List[int],
                 gamma: float):
        super().__init__(optimizer)
        self.milestones = np.array(milestones)
        self.gamma = gamma
        
        self.max_learning_rate = optimizer.get_initial_learning_rate()
    
    
    def get_learning_rate_at_T(self, T: int):
        milestones_passed = (T >= self.milestones).sum()
        return self.max_learning_rate * self.gamma ** milestones_passed


class CosineAnnealingLR(Scheduler):
    """Decay the learning rate according to a cosine curve."""
    
    def __init__(self,
                 optimizer,
                 T_max: int,
                 min_learning_rate: float):
        super().__init__(optimizer)
        self.T_max = T_max
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = optimizer.get_initial_learning_rate()
    
    
    def get_learning_rate_at_T(self, T: int):
        return (
            self.min_learning_rate
            + 0.5 * (self.max_learning_rate - self.min_learning_rate) * (1 + np.cos(np.pi * T / self.T_max))
        )

    
class WarmupWrapper(Scheduler):
    """Wrapper around another scheduler. Linear warmup for the first `warmup_steps`, then uses the wrapped scheduler."""
    
    def __init__(self,
                 scheduler,
                 warmup_steps: int):
        super().__init__()
        scheduler.optimizer.scheduler = self  # Override optimizer's scheduler

        self.warmup_steps = warmup_steps
        self.scheduler = scheduler
        
        self.max_learning_rate = scheduler.optimizer.get_initial_learning_rate()
    
    
    def get_learning_rate_at_T(self, T: int):
        if self.T <= self.warmup_steps:
            return self.T / self.warmup_steps * self.max_learning_rate
        else:
            return self.scheduler.get_learning_rate_at_T(self.T - self.warmup_steps)
