"""Utils for generating text from a language model."""

import time
import re
import pygments
import numpy as np
from typing import Dict
from IPython.display import clear_output

import candle


def interactive_conversation(model,
                             chat_template,
                             tokenizer,
                             user_name: str,
                             profile_pic: str = '🙂',
                             user_bg_color: str = 'yellow',
                             asst_name: str = 'LLM',
                             asst_profile_pic: str = '🤖',
                             asst_bg_color: str = 'black',
                             max_response_length: int = 100,
                             top_k: int = 100,
                             top_p: float = 0.98,
                             temperature: float = 0.8,
                             stop_token_idx: int = None,
                             stop_strings: dict = None,
                             use_kv_cache: bool = True):
    """Starts an interactive conversation in Jupyter notebook.
    
    Note: if cells start auto-collapsing, this is an issue in Jupyter 7. Use nbclassic to fix.
    """
    stdout = StdoutWithSyntaxHighlighting()
    
    user_pic = ansi_color(profile_pic, 'bright', bg_color=user_bg_color) + ansi_color(f' {user_name}:', 'bright')
    asst_pic = ansi_color(asst_profile_pic, 'bright', bg_color=asst_bg_color) + ansi_color(f' {asst_name}:', 'bright')

    stdout.print(ansi_color(
        f'< You are now talking with {asst_name}. Send \'bye\' to exit, \'clear\' to reset cache. >',
        style='bright'
    ), end='')

    if use_kv_cache:
        model.clear_kv_cache()
    messages = [{'role': 'system', 'content': chat_template.system_message}]
    last_chat = ''
    while True:
        stdout.print('\n\n' + user_pic, end=' ')
        time.sleep(0.2)  # Sometimes the input() doesn't show if we don't add a delay
        prompt = input()
        stdout.print(prompt)

        if prompt.lower().strip() == 'bye':
            stdout.print(ansi_color(f'\n< / end of conversation. >', style='bright'))
            return
        elif prompt.lower().strip() == 'clear' and use_kv_cache:
            stdout.print(ansi_color(f'< Cache cleared >', style='bright', color='white'), end='')
            model.clear_kv_cache()
            messages = [{'role': 'system', 'content': chat_template.system_message}]
            last_chat = ''
            continue

        messages.append({'role': 'user', 'content': prompt})
        chat = chat_template.apply_chat_template(messages, add_generation_prompt=True)
        if use_kv_cache:
            chat_update = chat[len(last_chat):]  # Feed only chat update into model because we use KV caching
        else:
            chat_update = chat
            
        stdout.print('\n' + asst_pic, end=' ')
        response = generate_text(
            model,
            tokenizer,
            prompt=chat_update,
            n_tokens_to_gen=max_response_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            stop_token_idx=stop_token_idx,
            stop_strings=stop_strings,
            use_kv_cache=use_kv_cache,
            stdout=stdout,
        )
        messages.append({'role': 'assistant', 'content': response})
        last_chat = chat + response


def generate_text(model,
                  tokenizer,
                  prompt: str,
                  n_tokens_to_gen: int,
                  top_k: int = 40,
                  top_p: int = 0.95,
                  temperature: float = 1.0,
                  sample: bool = True,
                  stop_token_idx: int = None,
                  stop_strings: Dict[str, int] = None,
                  print_stream: bool = True,
                  use_kv_cache: bool = True,
                  stdout = None):
    """Given a conditioning prompt, generates N tokens using beam search.
    
    Args:
        n_tokens_to_gen (int): Number of tokens to generate.
        top_k (float): Filter probabilities to those in the top k.
        top_p (float): Nucleus sampling. Filter to top probs such that the sum is just less than top_p.
        temperature (float): Higher temperature raises the likelihood of lower probability sequences.
        sample (bool): True to randomly sample sequences from the distribution of probabilities
            False to take argmax.
        stop_token_idx (int): If provided, terminates generation upon seeing `stop_token_id`.
        stop_strings: Dict mapping string to how many times we can see the string before
            we stop generation. Accepts regexes.
            
            Example:
                stop_strings = {
                    f'John:': 1,
                    f'Taylor:': 1,
                    '\n': 1,
                    '<|endoftext|>': 1,
                    '\.|\!|\?': 4      # If we see a . or ! or ? more than 4 times total
                }
        print_stream (bool): If True, then prints tokens as they are generated.
        use_kv_cache (bool):            If True, uses KV caching to speed up inference.
        stdout: Object implementing stdout.print(...). If None, prints to sys.stdout.
        
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
            if stop_token_idx in indices_to_decode:
                stop_gen = True
                indices_to_decode = indices_to_decode[:indices_to_decode.index(stop_token_idx)]
        
            token = tokenizer.decode(indices_to_decode)
            indices_to_decode = []
            token = ''.join(token)
            
            # If we see any regex `s` at least `stop_strings[s]` times, end generation
            for s in stop_strings:
                matches = list(re.finditer(s, response + token))
                if len(matches) >= stop_strings[s]:
                    (left_index, right_index) = matches[-1].span()

                    token = (response + token)[len(response):left_index]
                    stop_gen = True
                    
            if response == '':
                token = token.lstrip()
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

    Args:
        text (str): The text to be formatted.
        style (str, optional): The desired text style. Valid options are:
            * 'bright'
            * 'dim'
            * 'underscore'
            * 'blink'
            * 'reverse'
            * 'hidden'
        color (str, optional): The desired text color. Valid options are:
            * 'black'
            * 'red'
            * 'green'
            * 'yellow'
            * 'orange'
            * 'blue'
            * 'magenta'
            * 'cyan'
            * 'white'
            * 'bluegreen'
        bg_color (str, optional): The desired background color. Valid options are the same
            as for `color`.
    
    Returns:
        str. The formatted text with ANSI escape sequences added.

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
            orange='\x1b[33m',
            blue='\x1b[34m',
            magenta='\x1b[35m',
            cyan='\x1b[36m',
            white='\x1b[37m',
        ),
        'bg_color': dict(
            black='\x1b[40m',
            red='\x1b[41m',
            green='\x1b[42m',
            yellow='\x1b[43m',
            orange='\033[48;5;202m',
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
    
    Example:
        stdout = StdoutWithSyntaxHighlighting()
        stdout.print('Hi this is code: ```def f(x): return x```')


    Warning:
        Clears entire output of cell every time .print() is called.
        Weird cell collapsing happens in the latest version of Jupyter Notebook 7.
        Use nbclassic to fix.
    
    """
    
    def __init__(self, code_delim: str = '```python|```'):
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
    