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
from keras_nlp.tokenizers import BytePairTokenizer

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras


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
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        vocab = {value: i for i, value in enumerate(vocab)}
        self.tokenizer = BytePairTokenizer(
            vocabulary=vocab,
            merges=merges,
        )

    def encode(self, text):
        return self.tokenizer.tokenize(text)

    def decode(self, tokens):
        return self.tokenizer.detokenize(tokens)

    def get_id(self, token):
        return self.tokenizer.token_to_id(token)

    def get_token(self, id):
        return self.tokenizer.id_to_token(id)
