# Copyright 2023 The KerasCV Authors
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
import regex as re
import tensorflow as tf

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None

try:
    import keras_nlp
    from keras_nlp.tokenizers import BytePairTokenizer
except ImportError:
    keras_nlp = None
    BytePairTokenizer = object

# As python and TF handles special spaces differently, we need to
# manually handle special spaces during string split.
SPECIAL_WHITESPACES = r"\x{a0}\x{2009}\x{202f}\x{3000}"
SPLIT_PATTERN_1 = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    + r"|[\s{special_spaces}]+[\n\r\t\f६{special_spaces}]| ?\p{L}+|"
    + r" ?[\p{N}]+| ?[^\s\p{L}\p{N}{special_spaces}</w>]+"
)
SPLIT_PATTERN_1 = SPLIT_PATTERN_1.replace(
    "{special_spaces}", SPECIAL_WHITESPACES
)
SPLIT_PATTERN_2 = rf"""[\s६{SPECIAL_WHITESPACES}]$"""


def split_strings_for_bpe(inputs, unsplittable_tokens=None):
    # We need to recreate the exact behavior of token presplitting in the
    # original gpt2 tokenizer which uses a lookahead. As re2 does not
    # support lookahead match, we are using an alternative insert a special
    # token "६" before leading space of non-space characters and after the
    # trailing space, e.g., " keras" will be "६ keras".
    if tf_text is None:
        raise ImportError(
            "BytePairTokenization requires `tensorflow_text`."
            "Please install with `pip install tensorflow_text`."
        )
    inputs = tf.strings.regex_replace(
        inputs, rf"( )([^\s{SPECIAL_WHITESPACES}])", r"६\1\2"
    )
    inputs = tf.strings.regex_replace(
        inputs, rf"(\s{SPECIAL_WHITESPACES})$", r"\1६"
    )
    inputs = tf.strings.regex_replace(inputs, r"\s", "")
    if unsplittable_tokens:
        alts = create_alts_for_unsplittable_tokens(unsplittable_tokens)
        for token, alt in zip(unsplittable_tokens, alts):
            escaped_token = re.escape(token)
            inputs = tf_text.regex_split(inputs, escaped_token, escaped_token)
            inputs = tf.strings.regex_replace(inputs, escaped_token, alt)
    raw_tokens = tf_text.regex_split(inputs, SPLIT_PATTERN_1, SPLIT_PATTERN_1)
    # Second pass splits out the last whilespace char or "६".
    raw_tokens = tf_text.regex_split(
        raw_tokens, SPLIT_PATTERN_2, SPLIT_PATTERN_2
    )
    if unsplittable_tokens:
        # Replace special tokens alternate with originals.
        for token, alt in zip(unsplittable_tokens, alts):
            escaped_alt = re.escape(alt)
            raw_tokens = tf.strings.regex_replace(
                raw_tokens, escaped_alt, token
            )

    # Add '</w>' to the end of each token
    tokens_with_end_tag = tf.strings.regex_replace(
        raw_tokens, r"(\p{L}+)", r"\1</w>"
    )

    while tokens_with_end_tag.shape.rank > 2:
        tokens_with_end_tag = tokens_with_end_tag.merge_dims(1, 2)

    return remove_strings_from_inputs(tokens_with_end_tag, "६")


def create_alts_for_unsplittable_tokens(unsplittable_tokens):
    # Create alternates for all special tokens that will be not split during
    # tokenization.
    alts = []
    prefix = "Ĵ"
    # Trim out splitters.
    replace_pattern = r"'|\s+|[^\p{L}\p{N}]+"
    for token in unsplittable_tokens:
        token = re.sub(replace_pattern, "", token)
        alts.append(prefix + token)
    return alts


def remove_strings_from_inputs(tensor, string_to_remove):
    """Remove certain strings from input tensor."""
    non_empty_mask = tensor != string_to_remove
    flatten_indexes = tf.where(non_empty_mask)
    flatten_result = tf.gather_nd(tensor, flatten_indexes)
    row_lengths = tf.reduce_sum(tf.cast(non_empty_mask, "int64"), axis=1)
    result = tf.RaggedTensor.from_row_lengths(
        values=flatten_result,
        row_lengths=row_lengths,
    )
    return result


class CLIPTokenizer(BytePairTokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if keras_nlp is None:
            raise ValueError(
                "ClipTokenizer requires keras-nlp. Please install "
                "using pip `pip install -U keras-nlp && pip install -U keras`"
            )

    def _bpe_merge_and_update_cache(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._transform_bytes(tokens)
        tokenized_words = self._bpe_merge(words)

        # For each word, join all its token by a whitespace,
        # e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words,
            axis=1,
        )
        self.cache.insert(tokens, tokenized_words)

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        if self.add_prefix_space:
            inputs = tf.strings.join([" ", inputs])

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        raw_tokens = split_strings_for_bpe(inputs, self.unsplittable_tokens)
        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values
        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""

        has_unseen_words = tf.math.reduce_any(
            (cache_lookup == "") & (flat_tokens != "")
        )

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )
        tokens = tf.strings.split(tokenized_words, sep=" ")
        if self.compute_dtype != tf.string:
            # Encode merged tokens.
            tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input in scalar
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens
