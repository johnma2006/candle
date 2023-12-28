from typing import List
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """An encoder that encodes/decodes text to and from integer tokens."""

    def __init__(self):
        pass
    
    
    @abstractmethod
    def encode(self, text: str):
        """Encodes text into list of integers.
        
        Args:
            text (str): String to encode.
            
        Returns:
            indices: List of integers representing encoded string.
            
        """
        pass
    

    @abstractmethod
    def decode(self, indices: List[int]):
        """The inverse of encode(). Decodes list of integers into text.
        
        Args:
            indices: List of integers representing encoded string.
            
        Returns:
            text: String to encode.
            
        """
        pass
