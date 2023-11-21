import numpy as np
from typing import List
import unittest

import torch
import torch.nn as nn

import candle as candle
import candle.functions as F
import candle.optimizer


class TestTokenizer(unittest.TestCase):
    
    def test_tokenizer_encode_decode(self):
        tokenizer = candle.models.gpt.GPT2BPETokenizer()
        for tokenizer in range(100):
            text = ''.join([chr(i) for i in np.random.choice(range(10000), size=10)])
            assert text == self.decode(self.encode(text))
