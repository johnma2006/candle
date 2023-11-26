"""Utils for generating text from a language model."""

import numpy as np

import candle


def generate_text(model,
                  tokenizer,
                  prompt: str,
                  n_tokens_to_generate: int,
                  beam_size: int = 1,
                  top_k: int = 40,
                  top_p: int = 0.95,
                  temperature: float = 1.0,
                  sample: bool = True,
                  end_of_text_token: str = '<|endoftext|>'):
    """Given a conditioning prompt, generates N more tokens using beam search.
    
    Returns a generator that yields tokens as they are generated.
    
    Parameters
    ----------
    n_tokens_to_generate
        Number of tokens to generate.
    beam_size
        Beam size to use in beam search. e.g., beam_size = 1 for greedy search.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum prob is just less than top_p.

        References:
        [1] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.
            The Curious Case of Neural Text Degeneration. arXiv:1904.09751, 2019
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities, False to take argmax.
    end_of_text_token
        If provided, seeds empty generaetion with `end_of_text_token`.
    
    """
    if prompt == '':
        indices = candle.Tensor(np.array([tokenizer.token_to_index[end_of_text_token]]))
    else:
        indices = candle.Tensor(np.array(tokenizer.encode(prompt)))

    generator = candle.nlp.beam_search_decoder(model, indices,
                                               n_tokens_to_generate=n_tokens_to_generate,
                                               beam_size=beam_size,
                                               top_k=top_k,
                                               top_p=top_p,
                                               temperature=temperature,
                                               sample=sample)

    indices_to_decode = np.array([])
    for next_indices in generator:
        indices_to_decode = np.concatenate([indices_to_decode, next_indices])
        try:
            token = tokenizer.decode(indices_to_decode)
            token = ''.join(token)
            yield token
            indices_to_decode = np.array([])
        except GeneratorExit:
            return
        except:
            # Sometimes, we can't decode properly until we get a few more indices (e.g. if it falls
            # between utf-8 byte boundaries). We accumuate those indices in `indices_to_decode` until
            # it becomes decodable
            pass

        
def display_stream_of_text(model,
                           tokenizer,
                           prompt: str,
                           n_tokens_to_generate: int,
                           beam_size: int = 5,
                           top_k: int = 40,
                           top_p: int = 0.95,
                           temperature: float = 1.0,
                           sample: bool = True,
                           stop_strings: bool = True):
    """Given a conditioning prompt, generates N more tokens using beam search.
    
    Prints the text as it is generated.
    
    Parameters
    ----------
    n_tokens_to_generate
        Number of tokens to generate.
    beam_size
        Beam size to use in beam search. e.g., beam_size = 1 for greedy search.
    top_k
        Filter probabilities to those in the top k.
    top_p
        Nucleus sampling. Filter to top probs such that the sum prob is just less than top_p.

        References:
        [1] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi.
            The Curious Case of Neural Text Degeneration. arXiv:1904.09751, 2019
    temperature
        Higher temperature raises the likelihood of lower probability sequences.
    sample
        True to randomly sample sequences from the distribution of probabilities, False to take argmax.
    stop_strings
        If True, then stops text generation as soon as we see one of these strings.
    
    """
    generator = generate_text(model, tokenizer, prompt, n_tokens_to_generate,
                              beam_size, top_k, top_p, temperature, sample)
    
    response = ''
    for (tokens_generated, token) in enumerate(generator):
        # If first token, strip leading whitespace
        if tokens_generated == 0:
            token = token.strip()
            
        # If we see any string in `stop_strings`, end generation
        stop_generation = False
        for s in stop_strings:
            if s in response + token:
                index = (response + token).index(s)
                token = (response + token)[len(response):index]
                stop_generation = True

        response += token
        print(token, end='')
        
        if stop_generation:
            break
    
    return response
