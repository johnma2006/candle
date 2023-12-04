import numpy as np
import unittest

import torch
from transformers import GPT2LMHeadModel

import candle
import candle.functions as F
from candle import Tensor
from candle.nlp.beamsearch import beam_search_decoder
from candle.models.gpt.model import GPT


class TestBeamSearch(unittest.TestCase):
    
    def test_beam_decoder_cumulative_kv_cache(self):
        """If KV caching, we will guarantee that:
               (kv_cache_seqlen after decoding) == (kv_cache_seqlen before decoding) + len(indices)
               - This will be true even if the generator terminates early, as long as generator.close() is called
        """
        for _ in range(4):

            model = GPT(
                n_layers=12,
                n_heads=8,
                embed_dim=32,
                vocab_size=200,
                block_size=128,
                dropout_p=0.1,
            )

            input1 = Tensor([0, 1, 2, 3, 5, 2, 3, 3])

            # Round 1

            initial_cache_seqlen = model.decoder_blocks[0].attn.get_kv_cache_seqlen()
            assert initial_cache_seqlen == 0

            generator = beam_search_decoder(
                model, 
                input1,
                n_tokens_to_generate=50,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=True,
                use_kv_cache=True,
            )


            for i in range(5):
                try:
                    token = next(generator)
                except StopIteration:
                    pass

            generator.close()
            assert model.decoder_blocks[0].attn.get_kv_cache_seqlen() == initial_cache_seqlen + len(input1)

            # Round 2

            generator = beam_search_decoder(
                model, 
                input1,
                n_tokens_to_generate=50,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=True,
                use_kv_cache=True,
            )

            initial_cache_seqlen = model.decoder_blocks[0].attn.get_kv_cache_seqlen()

            for i in range(5):
                try:
                    token = next(generator)
                except StopIteration:
                    pass

            generator.close()
            assert model.decoder_blocks[0].attn.get_kv_cache_seqlen() == initial_cache_seqlen + len(input1)

            # Round 3

            generator = beam_search_decoder(
                model, 
                input1,
                n_tokens_to_generate=50,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=True,
                use_kv_cache=True,
            )

            initial_cache_seqlen = model.decoder_blocks[0].attn.get_kv_cache_seqlen()

            while True:
                try:
                    token = next(generator)
                except StopIteration:
                    break

            generator.close()
            assert model.decoder_blocks[0].attn.get_kv_cache_seqlen() == initial_cache_seqlen + len(input1)


    def test_beam_decoder_cumulative_kv_cache_with_early_termination(self):

        for _ in range(4):
            # No KV cache

            model = GPT(
                n_layers=12,
                n_heads=8,
                embed_dim=32,
                vocab_size=200,
                block_size=128,
                dropout_p=0.1,
            )

            input1 = Tensor([0, 1, 2, 3, 5, 2, 3, 3])

            generator = beam_search_decoder(
                model, 
                input1,
                n_tokens_to_generate=50,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=False,
            )

            output1 = []
            for i in range(5):
                try:
                    output1.append(next(generator))
                except StopIteration:
                    pass
            output1 = Tensor(np.concatenate(output1))

            generator.close()

            input2 = Tensor([23, 54, 1, 3, 5, 2])

            generator = beam_search_decoder(
                model, 
                F.concat([input1, output1, input2]),
                n_tokens_to_generate=20,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=False,
            )

            output2_nokvcache = np.concatenate(list(generator))

            # Cumulative kv cache

            model.clear_kv_cache()
            input1 = Tensor([0, 1, 2, 3, 5, 2, 3, 3])

            generator = beam_search_decoder(
                model, 
                input1,
                n_tokens_to_generate=50,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=True,
            )

            output1 = []
            for i in range(5):
                try:
                    output1.append(next(generator))
                except StopIteration:
                    pass
            output1 = Tensor(np.concatenate(output1))

            generator.close()

            input2 = Tensor([23, 54, 1, 3, 5, 2])

            generator = beam_search_decoder(
                model, 
                F.concat([output1, input2]),
                n_tokens_to_generate=20,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=True,
            )

            output2_kvcache = np.concatenate(list(generator))

            assert np.alltrue(output2_kvcache == output2_nokvcache)


    def assert_beam_decoder_no_kv_cache_vs_kv_cache_equal(self):

        for _ in range(4):
            model = GPT(
                n_layers=12,
                n_heads=8,
                embed_dim=32,
                vocab_size=200,
                block_size=128,
                dropout_p=0.1,
            )

            indices = Tensor([0, 1, 2, 3, 5, 2, 3, 3])

            generator_no_kv_cache = beam_search_decoder(
                model, 
                indices,
                n_tokens_to_generate=20,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=False,
            )

            new_indices_no_kv_cache = np.concatenate(list(generator_no_kv_cache))

            generator_kv_cache = beam_search_decoder(
                model, 
                indices,
                n_tokens_to_generate=20,
                beam_size=3,
                top_k=40,
                top_p=0.95,
                sample=False,
                use_kv_cache=False,
            )

            new_indices_kv_cache = np.concatenate(list(generator_kv_cache))

            assert np.alltrue(new_indices_no_kv_cache == new_indices_kv_cache)