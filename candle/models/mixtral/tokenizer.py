"""Mistral's Mixtral tokenizer. The exact same as Llama's.

"""

import re
from typing import List

import candle
from candle.nlp.tokenizer import Tokenizer


class MixtralTokenizer(Tokenizer):
    
    def __init__(self,
                 tokenizer_model_path: str):
        """
        Parameters
        ----------
        tokenizer_model_path
            Path to tokenizer model, e.g. /path/to/tokenizer.model.
            
        """
        self.sp_model = candle.nlp.sentencepiece.Processor(tokenizer_model_path)
        
        
    def encode(self, text: str):
        """Encodes text into list of integers.
        
        Parameters
        ----------
        text
            String to encode.
            
        Returns
        -------
        indices
            List of integers representing encoded string.
            
        """
        encoded = []

        # We manually replace <s> and </s> with the correct bos/eos ids
        match = re.search('<s>|</s>', text)
        while match is not None:
            if match.group(0) == '<s>':
                encoded.append(self.sp_model.bos_id())
            else:
                encoded.append(self.sp_model.eos_id())

            encoded += self.sp_model.encode(text[:match.span()[0]])
            text = text[match.span()[1]:]
            match = re.search('<s>|</s>', text)

        encoded += self.sp_model.encode(text)
        
        return encoded
    

    def decode(self, indices: List[int], remove_leading_space: bool = False):
        """The inverse of encode(). Decodes list of integers into text.
        
        Parameters
        ----------
        indices
            List of integers representing encoded string.
            
        Returns
        -------
        text
            String to encode.
            
        """
        return self.sp_model.decode(indices, remove_dummy_prefix=remove_leading_space)
    