"""Operations on Tensors which define the forward and backward pass."""

from .operation import Operation
from .arithmetic import (
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Power,
    Exponentiation,
)
from .tensorops import (
    TensorContraction,
    TensorSum,
    TensorMax,
    TensorMin,
    TensorSlice,
    TensorSetSlice,
    TensorReshape,
    TensorSwapaxes,
    TensorTranspose,
    TensorConcatenation,
    TensorMaskedFill,
    TensorRepeatInterleave,
    BatchMatrixMultiply,
    TensorClone,
    TensorFlip,
    TopKOperation,
)
from .activations import (
    ReLUActivation,
    GeLUActivation,
    SiLUActivation,
    SoftplusActivation,
)
from .loss import CrossEntropyLossOperation
from .conv import (
    Conv2dOperation,
    MaxPool2dOperation,
    AvgPool2dOperation,
)
