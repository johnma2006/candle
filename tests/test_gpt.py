import numpy as np
from typing import List
import unittest

import torch
from transformers import GPT2LMHeadModel

import candle as candle


class TestGPT(unittest.TestCase):
    
    def test_equivalency_vs_hugging_face_impl(self):
        model = candle.models.gpt.GPT.from_pretrained('gpt2')
        model_hf = GPT2LMHeadModel.from_pretrained('gpt2')


        x = (100 * np.random.random(size=(16, 32))).astype(int)

        model.eval()
        model_hf.eval()

        logits = model(candle.Tensor(x)).data
        logits_hf = model_hf(torch.Tensor(x).int()).logits.detach().numpy()

        diff = logits - logits_hf
        ratio = logits / logits_hf

        assert -1e-3 < diff.min() < diff.max() < 1e-3
        assert 1 - 1e-5 < ratio.min() < ratio.max() < 1 + 1e-5
