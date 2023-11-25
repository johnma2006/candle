import numpy as np
import unittest

import candle


class TestTokenizer(unittest.TestCase):
    
    def test_tokenizer_encode_decode(self):
        tokenizer = candle.models.gpt.GPT2BPETokenizer()
        for _ in range(100):
            text = ''.join([chr(i) for i in np.random.choice(range(10000), size=10)])
            assert text == tokenizer.decode(tokenizer.encode(text))
