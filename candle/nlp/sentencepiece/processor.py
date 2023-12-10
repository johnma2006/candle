"""Lightweight Python implementation of SentencePiece encoding/decoding.

Should be similar to using Google's sentencepiece.SentencePieceProcessor, but with a lot
fewer features (just enough to encode/decode LLaMA's vocabulary).

References
----------
[1] https://github.com/google/sentencepiece/blob/master/src/sentencepiece_processor.cc

"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List
import heapq


class PieceType(IntEnum):
    NORMAL = 1        # Normal symbol.
    UNKNOWN = 2       # Unknown symbol. Only <unk> for now.
    CONTROL = 3       # Control symbols. </s>, <s>, <2ja> etc.
    USER_DEFINED = 4  # User defined symbols.
    BYTE = 6          # Byte symbols. Used when `byte_fallback` is true.
    UNUSED = 5        # This piece is not used.


class Processor:

    WHITESPACE_REPLACEMENT = '▁'  # If you squint you will notice this is not the same as '_'
                                  # '▁' has unicode ID ord('▁') = 9601  <-- what we use
                                  # '_' has unicode ID ord('_') = 95
        
    def __init__(self,
                 model_file: str):
        """Initialize Processor.
        
        Parameters
        ----------
        model_file
            Path to model file, e.g. /path/llama2/tokenizer.model.
            This file likely is the output of running Google's SentencePieceTrainer.
            
        """
        from . import spm_protobuf  # Need to `conda install protobuf`
        self.model_file = model_file
        
        self.model_proto = spm_protobuf.ModelProto()
        with open(model_file, 'rb') as f:
            self.model_proto.ParseFromString(f.read())
        
        # Invert score because we use min heaps
        self._score_by_token = {p.piece: -1.0 * p.score for p in self.model_proto.pieces}
        self._idx_by_token = {p.piece: idx for (idx, p) in enumerate(self.model_proto.pieces)}
        self._token_by_idx = {idx: p.piece for (idx, p) in enumerate(self.model_proto.pieces)}
        self._byte_pieces = [p.piece for p in self.model_proto.pieces if p.type == PieceType.BYTE]
        
        if self.normalizer_spec.remove_extra_whitespaces:
            raise NotImplementedError()  # Whitespace removal is not supported yet
        
        
    def encode(self,
               input_str: int,
               out_type: type = int,
               add_bos: bool = False,
               add_eos: bool = False,
               emit_unk_piece: bool = False):
        """O(nlogn) BPE merging algorithm.

        Parameters
        ---------
        input_str
            String to encode.
        out_type
            One of [str, int].
        add_bos
            Add <s> token to start.
        add_eos
            Add </s> token to end.
        emit_unk_piece
            Emit the UNK literal string.

        """
        if self.trainer_spec.model_type == self.trainer_spec.BPE:
            return self._encode_bpe(input_str, out_type, add_bos,
                                    add_eos, emit_unk_piece)
        
        elif self.trainer_spec.model_type == self.trainer_spec.UNIGRAM:
            raise NotImplementedError('Unigram encoding/decoding not implemented.')
            
        else:
            raise NotImplementedError()
    
        
    def decode(self, input_ids: List[int], remove_dummy_prefix: bool = True):
        """Decodes list of ids."""
        decoded = []
        byte_accumulation = []
        for i in input_ids:
            piece = self.id_to_piece(i)

            if self.get_piece_type(i) == PieceType.BYTE:
                byte_accumulation.append(self.id_to_byte(i))
            else:
                if len(byte_accumulation) > 0:
                    decoded.append(bytes(byte_accumulation).decode())
                    byte_accumulation = []

                if self.get_piece_type(i) == PieceType.NORMAL:
                    decoded.append(piece)
                elif self.get_piece_type(i) == PieceType.CONTROL:
                    decoded.append('')
                elif self.get_piece_type(i) == PieceType.UNKNOWN:
                    decoded.append(self.trainer_spec.unk_surface)

        if len(byte_accumulation) > 0:
            decoded.append(bytes(byte_accumulation).decode())

        decoded = ''.join(decoded)
        decoded = self._undo_preprocess_input(decoded, remove_dummy_prefix)

        return decoded

    
    def _encode_bpe(self,
                    input_str: int,
                    out_type: type = str,
                    add_bos: bool = False,
                    add_eos: bool = False,
                    emit_unk_piece: bool = False):
        """O(nlogn) BPE merging algorithm using priority queue"""
        assert out_type in [int, str]
        if input_str == '':
            return []
        
        # We initialize tokens as single characters, and form TokenPairs from
        # consecutive tokens and add to `token_pairs`, a min heap sorted by TokenPair.score.
        # Every iteration we take the TokenPair in `token_pairs` with lowest score and merge it.
        token_pairs = []
        
        @dataclass(order=True)
        class TokenPair:
            # Score is the BPE merging priority, defined in the model proto
            score: float
            left_token: str = field(compare=False)
            right_token: str = field(compare=False)

            # TokenPairs also define a doubly linked list which define its
            # relationship with other TokenPairs
            left_pair: TokenPair = field(compare=False)
            right_pair: TokenPair = field(compare=False)

            # Helper flag to deal with heap updating
            invalid: bool = False

            @staticmethod
            def copy(token_pair):
                """Return copy while maintaining linked-list pointers"""
                if token_pair is None:
                    return None
                else:
                    return TokenPair(token_pair.score,
                                     token_pair.left_token,
                                     token_pair.right_token,
                                     token_pair.left_pair,
                                     token_pair.right_pair,
                                     token_pair.invalid)

        input_str = self._preprocess_input(input_str)
        for (char1, char2) in zip(input_str, input_str[1:]):
            score = self.get_score(char1 + char2)
            token_pair = TokenPair(score, char1, char2,
                                   left_pair=None, right_pair=None)
            token_pairs.append(token_pair)
        
        # Attach doubly-linked list
        prev_pair = None
        for token_pair in token_pairs:
            token_pair.left_pair = prev_pair
            if prev_pair is not None:
                prev_pair.right_pair = token_pair
            prev_pair = token_pair
        
        # Turn token_pairs into a priority queue
        heapq.heapify(token_pairs)
        
        # Main loop: keep merging the lowest score pair
        while len(token_pairs) > 0 and token_pairs[0].score != float('inf'):  # token_pairs[0] is the top of heap
            candidate_pair = heapq.heappop(token_pairs)
            if candidate_pair.invalid:
                continue
            else:
                lowest_score_pair = candidate_pair
        
            # Merge pair, (b, c) -> bc
            merged_token = lowest_score_pair.left_token + lowest_score_pair.right_token
        
            # Update pairs to the left and right of (b, c)
            #     (a, b) (b, c) (c, d) -> (a, bc) (bc, d)
            # Since it's difficult to update elements in Python's heapq data struture, we do a workaround
            # by marking the old left/right as invalid and pop the new "updated" left/right
            left = TokenPair.copy(lowest_score_pair.left_pair)
            right = TokenPair.copy(lowest_score_pair.right_pair)
        
            if left is not None:
                left.score = self.get_score(left.left_token + merged_token)
                left.right_token = merged_token
                left.right_pair = right
                if left.left_pair is not None:
                    left.left_pair.right_pair = left
        
                lowest_score_pair.left_pair.invalid = True
                heapq.heappush(token_pairs, left)
        
            if right is not None:
                right.score = self.get_score(merged_token + right.right_token)
                right.left_token = merged_token
                right.left_pair = left
                if right.right_pair is not None:
                    right.right_pair.left_pair = right
        
                lowest_score_pair.right_pair.invalid = True
                heapq.heappush(token_pairs, right)
        
        if len(token_pairs) > 0:
            # Find leftmost pair
            pair = token_pairs[0]
            while pair.left_pair is not None:
                pair = pair.left_pair
            
            # Iterate through linked list, collecting the token strings
            merged_tokens = [pair.left_token, pair.right_token]
            while pair.right_pair is not None:
                pair = pair.right_pair
                merged_tokens.append(pair.right_token)
        
        else:  # This means everything was merged into one token == `lowest_score_pair`
            merged_tokens = [lowest_score_pair.left_token + lowest_score_pair.right_token]
        
        if add_bos:
            merged_tokens = [self.trainer_spec.bos_piece] + merged_tokens
        if add_eos:
            merged_tokens = merged_tokens + [self.trainer_spec.eos_piece]
        if self.trainer_spec.byte_fallback:
            merged_tokens = self._byte_fallback(merged_tokens)
        if out_type is int:
            merged_tokens = [self.piece_to_id(i) for i in merged_tokens]

        return merged_tokens

        
    def _preprocess_input(self, input_str: str):
        if self.normalizer_spec.add_dummy_prefix:
            input_str = ' ' + input_str

        input_str = input_str.replace(' ', self.WHITESPACE_REPLACEMENT)

        return input_str
    
    def _undo_preprocess_input(self, input_str: str, remove_dummy_prefix: bool = True):
        input_str = input_str.replace(self.WHITESPACE_REPLACEMENT, ' ')
        
        if self.normalizer_spec.add_dummy_prefix and remove_dummy_prefix:
            input_str = input_str[1:]

        return input_str


    def _byte_fallback(self, merged_tokens):
        fallback_tokens = []
        for token in merged_tokens:
            if token in self._score_by_token:
                fallback_tokens.append(token)
            else:
                token_as_byte = [self._byte_pieces[byte] for byte in bytearray(token.encode('utf-8'))]
                fallback_tokens += token_as_byte

        return fallback_tokens


    def vocab_size(self):
        return len(self._score_by_token)
    
    
    def bos_id(self):
        return self.trainer_spec.bos_id
    
    
    def eos_id(self):
        return self.trainer_spec.eos_id
    
    
    def pad_id(self):
        return self.trainer_spec.pad_id
    
    
    def id_to_piece(self, id: int):
        return self._token_by_idx.get(id)
    

    def id_to_byte(self, id):
        return self._byte_pieces.index(self.id_to_piece(id))

    
    def piece_to_id(self, token: str):
        return self._idx_by_token.get(token)
    
    
    def get_score(self, token: str):
        return self._score_by_token.get(token, float('inf'))
    
    
    def get_piece_type(self, id: int):
        """Gets piece type, see PieceType enum."""
        return self.model_proto.pieces[id].type
    
    
    @property
    def trainer_spec(self):
        return self.model_proto.trainer_spec

    
    @property
    def normalizer_spec(self):
        return self.model_proto.normalizer_spec
    