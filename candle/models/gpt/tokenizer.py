"""OpenAI's GPT2 tokenizer.

References:
[1] https://github.com/openai/gpt-2/blob/master/src/encoder.py

"""

import regex
import json
from typing import List

import candle
from candle.nlp.tokenizer import Tokenizer
from candle.nlp import bpe


class GPT2BPETokenizer(Tokenizer):
    
    def __init__(self):
        (token_to_index, merges) = GPT2BPETokenizer.download_vocab_from_openai()
        
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for (k, v) in token_to_index.items()}

        self.merges = merges
        
        self.byte_to_unicode_char = get_byte_to_unicode_char()
        self.unicode_char_to_byte = {v: k for (k, v) in self.byte_to_unicode_char.items()}

        # For an explanation of this regex, see github.com/karpathy/minGPT/blob/master/mingpt/bpe.py#L76
        self.pretokenize_regex = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        
        
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
        indices = []

        for word in self.pretokenization(text):
            # Example, word = ' 😊.'
            word_bytes = word.encode('utf-8') # word_bytes = b' \xf0\x9f\x98\x8a.'
            word_unicode = ''.join([self.byte_to_unicode_char[byte] for byte in word_bytes])  # word_unicode = 'ĠðŁĺĬ.'
            word_tokenized = bpe.tokenize(word_unicode, self.merges)  # word_tokenized = ['ĠðŁĺ', 'Ĭ', '.']
            word_indices = [self.token_to_index[token] for token in word_tokenized]  # word_indices = [30325, 232, 13]
            indices += word_indices

        return indices


    def decode(self, indices: List[int]):
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
        unicode_repr = ''.join([self.index_to_token[index] for index in indices])
        byte_repr = [self.unicode_char_to_byte[unicode_char] for unicode_char in unicode_repr]
        
        return bytearray(byte_repr).decode('utf-8')


    def pretokenization(self, text: str):
        """Splits text into list of words.
        
        Example: 
            > pretokenization("Hi, my name is John and I'm happy 😊.")
            >> ['Hi', ',', ' my', ' name', ' is', ' John', ' and', ' I', "'m", ' happy', ' 😊.']
        
        """
        return regex.findall(self.pretokenize_regex, text)
    
    
    @staticmethod
    def download_vocab_from_openai():
        """Downloads the pre-trained BPE vocab and merges from OpenAI."""
        token_to_index = candle.utils.download_and_cache_file(
            'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json',
            cache_file_name='gpt2_encoder.json'
        )
        token_to_index = json.loads(token_to_index)

        merges = candle.utils.download_and_cache_file(
            'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe',
            cache_file_name='gpt2_vocab.bpe',
            encoding='utf-8'
        )
        merges = [tuple(line.split()) for line in merges.split('\n')[1:-1]]

        return (token_to_index, merges)

    
# From https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
def get_byte_to_unicode_char():
    """Returns dictionary mapping utf-8 byte to unicode character.
    
    The logic is roughly:
    
        for i in range(256)
            if chr(i) is a "single-char"-looking unicode, e.g. '!', '#', 'a', '3', '§', '©'
                bytes_to_unicode[i] = chr(i)
            otherwise if chr(i) looks like e.g. '\x1c', '\x7f', '\x80', '\x84'
                bytes_to_unicode[i] = chr(i + 256)  <-- this will happen to be a single-char looking unicode
          
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
