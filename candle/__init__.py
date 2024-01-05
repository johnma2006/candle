import os
# Workaround for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  

from .tensor import (
    Tensor,
    Parameter,
    rand,
    randn,
    zeros_like,
    ones_like,
    empty_like,
)
from .layers import (
    Module,
    ParameterList,
    Linear,
    Embedding,
    Dropout,
    BatchNorm,
    LayerNorm,
    RMSNorm,
    GroupedQueryRotaryAttention,
    MultiheadAttention,
    DotProductAttention,
    PositionalEncoding,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
)
from .dataloader import DataLoader, TokenDataLoader
from .tensorboard import Dashboard

from . import optimizer
from . import scheduler
from . import vision
from . import nlp
from . import models
from . import weightinit as init


# ------------------------------------
# Global is_grad_enabled functionality
# ------------------------------------

IS_GRAD_ENABLED = True

class set_grad_enabled:
    """Context manager that sets grad enabled on or off globally
    
    Examples
    --------
    with candle.set_grad_enabled(False):
        print(candle.is_grad_enabled())  # False
    print(candle.is_grad_enabled())  # True
    
    candle.set_grad_enabled(False)
    print(candle.is_grad_enabled())  # False
    
    """
    def __init__(self, mode):
        global IS_GRAD_ENABLED
        self.prev_is_grad_enabled = IS_GRAD_ENABLED
        IS_GRAD_ENABLED = mode
        
        
    def __enter__(self):
        pass
        

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global IS_GRAD_ENABLED
        IS_GRAD_ENABLED = self.prev_is_grad_enabled
        
        
class no_grad:
    """Context manager that disables grad.
    
    Examples
    --------
    with candle.no_grad(False):
        print(candle.is_grad_enabled())  # False
    print(candle.is_grad_enabled())  # True
    
    candle.no_grad(False)
    print(candle.is_grad_enabled())  # True
    
    """
    def __enter__(self):
        global IS_GRAD_ENABLED
        self.prev_is_grad_enabled = IS_GRAD_ENABLED
        IS_GRAD_ENABLED = False
        

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global IS_GRAD_ENABLED
        IS_GRAD_ENABLED = self.prev_is_grad_enabled
        
        
def is_grad_enabled():
    global IS_GRAD_ENABLED
    return IS_GRAD_ENABLED
