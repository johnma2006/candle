"""Byte-Pair Encoding tokenization."""

import numpy as np
from typing import List, Tuple, Dict


def tokenize(word: str,
             merges: List[Tuple[str, str]],
             merge_order: Dict[Tuple[str, str], int] = None,
             verbose: bool = False):
    """Given a word, breaks the word up into tokens using the BPE merges.
    
    Parameters
    ----------
    word
        String to tokenize.
    merges
        List of BPE merge pairs, the result of the BPE algorithm. The word will be merged according
        to the token pairs in `merges`, where earlier indices take priority.
        
        Example:
            merges = [('t', 'a'), ('d', 'o'), ('o', 'b'), ('n', 'o'), ('do', 'o'), ('doo', 'r'), ...]
            
            If word = 'doorknob', then word_tokens is initialized as ['d', 'o', 'o', 'r', 'k', 'n', 'o', 'b']
                (1) Merging ('d', 'o') gets us ['do', 'o', 'r', 'k', 'n', 'o', 'b']
                (2) Merging ('o', 'b') gets us ['do', 'o', 'r', 'k', 'n', 'ob']
                (3) Merging ('do', 'o') gets us ['doo', 'r', 'k', 'n', 'ob']
                (4) Merging ('doo', 'r') gets us ['door', 'k', 'n', 'ob']
                (5) ...
    merge_order
        This is equivalent to dict(zip(merges, range(len(merges)))). Pass this in to speed things up a bit.
    verbose
        If True, then prints out intermediate `word_tokens` for visualization/debugging.
    
    Returns
    -------
    word_tokens
        Tokenized word, list of str tokens which satisfies word == ''.join(word_tokens)
    
    """
    # Initialize word_tokens as each individual character
    word_tokens = [char for char in word]
    if verbose:
        print(f'word_tokens = {word_tokens}')

    # Create dict mapping each pair in `merges` to its index in `merges`. Lower index pairs will take priority later
    if merge_order is None:
        merge_order = dict(zip(merges, range(len(merges))))

    # Keep merging token pairs in word_tokens based on `merges` until we can't merge anymore
    while len(word_tokens) > 1:
        token_pairs = list(zip(word_tokens, word_tokens[1:]))

        # Find which pair in token_pairs appears first in `merges`
        highest_priority_pair = token_pairs[np.argmin([merge_order.get(pair, np.inf) for pair in token_pairs])]

        if highest_priority_pair not in merges: 
            # This means that no pair in token_pairs appear in `merges`, so we're done
            break

        # Merge all occurrences of highest_priority_pair in word_tokens
        
        word_tokens = merge_tokens(word_tokens, highest_priority_pair)
        if verbose:
            print(f'Merged {highest_priority_pair}. word_tokens = {word_tokens}')
            
    assert word == ''.join(word_tokens)
    return word_tokens


def generate_byte_pair_encoding(corpus: List[str],
                                num_merges: int):
    """Generates tokenization scheme using the Byte-Pair Encoding algorithm.
    
    Parameters
    ----------
    corpus
        List of words representing the full corpus to train the BPE algorithm on.
        e.g., ['To', ' be', ' or', ' not', ' to', ' be', ',', ' that', ...]
    num_merges
        Number of BPE merges to do.
    
    Returns
    -------
    (vocab, merges)
        `vocab` is a list of strings representing the vocabulary for the generated tokenization scheme.
        `merges` is the list of pairs of tokens which will be used to tokenize new words.
            e.g. merges = (('t', 'a'), ('d', 'o'), ('o', 'b'), ('n', 'o'), ('do', 'o'), ('doo', 'r'), ...)
    
    """
    # Map from word to how many times it appears in the corpus
    word_freqs = {}
    for word in corpus:
        if word not in word_freqs:
            word_freqs[word] = 0
        word_freqs[word] += 1 

    # Initialize vocabulary as the set of characters in all words
    vocab = set()
    for word in word_freqs:
        vocab |= set(word)
    vocab = sorted(list(vocab))

    # word_tokens maps each word to the current tokenization of that word
    word_tokens = {word: list(word) for word in word_freqs}

    merges = []
    for _ in range(num_merges):
        # Find highest frequency occuring pair

        pair_freqs = {}
        for word in word_freqs:
            for pair in zip(word_tokens[word], word_tokens[word][1:]):
                if pair not in pair_freqs:
                    pair_freqs[pair] = 0
                pair_freqs[pair] += word_freqs[word]

        if len(pair_freqs) == 0:
            # This means that there are no more pairs to merge
            break

        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)

        # Update vocab and merges

        vocab.append(most_frequent_pair[0] + most_frequent_pair[1])
        merges.append(most_frequent_pair)

        # Update split_words by applying merge using `most_frequent_pair`

        for word in word_tokens:
            word_tokens[word] = merge_tokens(word_tokens[word], most_frequent_pair)
    
    return (vocab, merges)


def merge_tokens(word_tokens: List[str], pair: Tuple[str, str]):
    """Merges list of tokens using a given merge pair.
    
    Example:
        > merge_tokens(['s', 'm', 'a', 'r', 't', 'a', 'r'], ('a', 'r'))
        >> ['s', 'm', 'ar', 't', 'ar']
    
    """
    new_word_tokens = []
    
    i = 0
    while i < len(word_tokens):
        if word_tokens[i] == pair[0] and (i + 1) < len(word_tokens) and word_tokens[i + 1] == pair[1]:
            new_word_tokens.append(pair[0] + pair[1])
            i += 1
        else:
            new_word_tokens.append(word_tokens[i])
        i += 1

    return new_word_tokens
