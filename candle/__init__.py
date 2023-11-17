import os
# Workaround for OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  

from . import optimizer
from . import scheduler

from .tensor import Tensor
from .parameter import Parameter, ParameterList
from .dataloader import DataLoader
from .tensorboard import Dashboard

from .layers.module import Module
from .layers.linear import Linear
from .layers.conv import (
    Conv2d,
    MaxPool2d,
    AvgPool2d,
)
from .layers.normalization import (
    BatchNorm,
    LayerNorm,
)