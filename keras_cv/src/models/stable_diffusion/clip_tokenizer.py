# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This code is taken nearly verbatim from
https://github.com/divamgupta/stable-diffusion-tensorflow."""

import gzip
import html
from functools import lru_cache

import regex as re

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@lru_cache()
def bytes_to_unicode():
    """Return a list of utf-8 bytes and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you
    want to avoid UNKs. When you're at something like a 10B token dataset you
    end up needing around 5K for decent coverage. This is a significant
    percentage of your normal, say, 32K bpe vocab. To avoid that, we want
    lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    A word is represented as tuple of symbols(symbols being variable-length
    strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


@keras_cv_export("keras_cv.models.stable_diffusion.SimpleTokenizer")
class SimpleTokenizer:
    def __init__(self, bpe_path=None):
        bpe_path = bpe_path or keras.utils.get_file(
            "bpe_simple_vocab_16e6.txt.gz",
            "https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true",  # noqa: E501
            file_hash="924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a",  # noqa: E501
        )
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.vocab = vocab
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.special_tokens = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = self._create_pat()

    def _create_encoder(self, vocab):
        return dict(zip(vocab, range(len(vocab))))

    def _create_decoder(self, encoder):
        return {v: k for k, v in encoder.items()}

    def _create_pat(self):
        return re.compile(
            "|".join([re.escape(key) for key in self.special_tokens.keys()])
            + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    @property
    def end_of_text(self):
        return self.encoder["<|endoftext|>"]

    @property
    def start_of_text(self):
        return self.encoder["<|startoftext|>"]

    def add_tokens(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        tokens_added = 0
        for token in tokens:
            if token in self.vocab:
                continue
            tokens_added += 1
            self.vocab.append(token)
            self.special_tokens[token] = token
            self.cache[token] = token
        self.encoder = self._create_encoder(self.vocab)
        self.decoder = self._create_decoder(self.encoder)
        self.pat = self._create_pat()
        return tokens_added

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if (
                    word[i] == first
                    and i < len(word) - 1
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(" ")
            )
        return [self.start_of_text] + bpe_tokens + [self.end_of_text]

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text
