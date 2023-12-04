"""Functionality to load LLaMA2 with Meta's pre-trained weights"""

import numpy as np
import transformers

from candle.tensor import Tensor


def load_pretrained_llama(model_name: str):
    """Returns LLaMA2 with pretrained weights.

    Parameters
    -----------
    model_name
        One of ['todo']
            todo:         124M params

    Returns
    -------
    model
        LLaMA2 instance with pre-trained weights initialized.

    """
#     todo
