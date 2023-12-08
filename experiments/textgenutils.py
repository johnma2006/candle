"""Utils for generating text from a language model."""

import numpy as np
import re
from typing import Dict
import pygments
from IPython.display import clear_output

import candle


def generate_text(model,
                  tokenizer,
                  prompt: str,
                  n_tokens_to_gen: int,
                  top_k: int = 40,
                  top_p: int = 0.95,
                  temperature: float = 1.0,
                  sample: bool = True,
                  stop_gen_token_idx: int = None,
                  stop_strings: Dict[str, int] = None,
                  print_stream: bool = True,
                  use_kv_cache: bool = True,
                  stdout = None):
    """Given a conditioning prompt, generates N tokens using beam search.
    
    Parameters
    ----------
    n_tokens_to_gen
        Number of tokens to generate.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities
        False to take argmax.
    stop_gen_token_idx
        If provided, terminates generation upon seeing `stop_gen_token_id`.
    stop_strings
        Dict mapping string to how many times we can see the string before
        we stop generation. Accepts regexes.
        
        Example:
            stop_strings = {
                f'John:': 1,
                f'Taylor:': 1,
                '\n': 1,
                '<|endoftext|>': 1,
                '\.|\!|\?': 4      # If we see a . or ! or ? more than 4 times total
            }
    print_stream
        If True, then prints tokens as they are generated.
    use_kv_cache
        If True, uses KV caching to speed up inference.
    stdout
        Object implementing stdout.print(...). If None, prints to sys.stdout.
    
    """
    if stop_strings is None:
        stop_strings = {}
        
    indices = candle.Tensor(np.array([tokenizer.encode(prompt)]))
    
    generator = candle.nlp.batch_generation(model, indices,
                                            n_tokens_to_gen=n_tokens_to_gen,
                                            top_k=top_k,
                                            top_p=top_p,
                                            temperature=temperature,
                                            sample=sample,
                                            use_kv_cache=use_kv_cache)

    stop_gen = False
    indices_to_decode = []
    response = ''
    for (tokens_generated, next_indices) in enumerate(generator):
        indices_to_decode = indices_to_decode + next_indices
        
        try:
            if stop_gen_token_idx in indices_to_decode:
                stop_gen = True
                indices_to_decode = indices_to_decode[:indices_to_decode.index(stop_gen_token_idx)]
        
            token = tokenizer.decode(indices_to_decode)
            indices_to_decode = []
            token = ''.join(token)

            if tokens_generated == 0:
                token = token.lstrip()
            
            # If we see any regex `s` at least `stop_strings[s]` times, end generation
            for s in stop_strings:
                matches = list(re.finditer(s, response + token))
                if len(matches) >= stop_strings[s]:
                    (left_index, right_index) = matches[-1].span()

                    token = (response + token)[len(response):left_index]
                    stop_gen = True
            
            response += token
            if print_stream:
                if stdout is None:
                    print(token, end='')
                else:
                    stdout.print(token, end='')
            
            if stop_gen:
                generator.close()  # Close generator to do any required KV cache cleanup
                return response
            
        except:
            # Sometimes, tokenizer.decode fails until we get a few more indices (e.g. if it falls
            # between utf-8 byte boundaries). We accumulate those indices in `indices_to_decode` until
            # it becomes decodable
            pass
        
    return response


def ansi_color(text: str,
               style: str = None,
               color: str = None,
               bg_color: str = None):
    """Formats text with ANSI escape codes for adding color and style.

    Parameters
    ----------
    text : str
        The text to be formatted.
    style : str, optional
        The desired text style. Valid options are:

        * 'bright'
        * 'dim'
        * 'underscore'
        * 'blink'
        * 'reverse'
        * 'hidden'
    color : str, optional
        The desired text color. Valid options are:

        * 'black'
        * 'red'
        * 'green'
        * 'yellow'
        * 'blue'
        * 'magenta'
        * 'cyan'
        * 'white'
        * 'bluegreen'
    bg_color : str, optional
        The desired background color. Valid options are the same as for
        `color`.

    Returns
    -------
    str
        The formatted text with ANSI escape sequences added.

    """
    RESET_CODE = "\x1b[0m"
    ANSI_CODES={
        'style': dict(
            bright='\x1b[1m',
            dim='\x1b[2m',
            underscore='\x1b[4m',
            blink='\x1b[5m',
            reverse='\x1b[7m',
            hidden='\x1b[8m',
        ),
        'color': dict(
            black='\x1b[30m',
            red='\x1b[31m',
            green='\x1b[32m',
            yellow='\x1b[33m',
            blue='\x1b[34m',
            magenta='\x1b[35m',
            cyan='\x1b[36m',
            white='\x1b[37m',
            bluegreen='\x1b[48;5;109m',
        ),
        'bg_color': dict(
            black='\x1b[40m',
            red='\x1b[41m',
            green='\x1b[42m',
            yellow='\x1b[43m',
            blue='\x1b[44m',
            magenta='\x1b[45m',
            cyan='\x1b[46m',
            white='\x1b[47m',
            bluegreen='\x1b[48;5;109m',
        )
    }

    for (setting, setting_str) in [(style, 'style'),
                                   (color, 'color'),
                                   (bg_color, 'bg_color')]:
        if setting is not None:
            if setting not in ANSI_CODES[setting_str]:
                raise ValueError(f'{setting_str} must be one of {list(ANSI_CODES[setting_str].keys())}.')
            text = ANSI_CODES[setting_str][setting] + text

    return text + RESET_CODE



class StdoutWithSyntaxHighlighting:
    """Emulates stdout printing, but with syntax highlighting.
    
    Example
    -------
    stdout = StdoutWithSyntaxHighlighting()
    stdout.print('Hi this is code: ```def f(x): return x```')


    Notes
    -----
    Warning: clears entire output of cell every time .print() is called.
    
    """
    
    def __init__(self, code_delim: str = '```'):
        self.code_delim = code_delim
        self.buffer = ''
        
        
    def print(self, *args, sep: str = ' ', end: str = '\n'):
        self.buffer += sep.join(args) + end
        clear_output(wait=True)
        print(self.highlight(self.buffer), end='')
        
    
    def highlight(self, text: str):
        """Returns `text` with syntax highlighted code blocks."""
        matches = list(re.finditer(self.code_delim, text))
        while len(matches) > 0:
            span0 = matches[0].span()
            if len(matches) >= 2:
                span1 = matches[1].span()
                code_block_end = ansi_color('</code>', color='white')
            else:  # text ends in code block
                span1 = (len(text) - 1, len(text))
                code_block_end = ''
                
            before_code = text[:span0[0]] + ansi_color('<code>', color='white')
            code = self.highlight_code_block(text[span0[1]:span1[0]])
            after_code = code_block_end + text[span1[1]:]

            text = before_code + code + after_code
            matches = list(re.finditer(self.code_delim, text))
            
        return text
    
    
    def highlight_code_block(self, code_block):
        return pygments.highlight(
            code_block,
            lexer=pygments.lexers.get_lexer_by_name('python', stripnl=False, ensurenl=False),
            formatter=pygments.formatters.Terminal256Formatter(style='default')
        )
    