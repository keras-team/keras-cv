# Copyright 2024 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gzip
import html
from functools import lru_cache

import regex as re

from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


def quick_gelu(x):
    return x * ops.sigmoid(1.702 * x)


class CLIPAttention(keras.layers.Layer):
    """
    Adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py # noqa: E501
    """

    def __init__(self, hidden_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.hidden_dim // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads (got `hidden_dim`"
                f": {self.hidden_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.dropout_layer = keras.layers.Dropout(self.dropout)
        self.scale = self.head_dim**-0.5
        self.query_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="query_proj",
        )
        self.key_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="key_proj",
        )
        self.value_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="value_proj",
        )
        self.out_proj = keras.layers.Dense(
            units=self.hidden_dim,
            name="out_proj",
        )

    def build(self, input_shape):
        self.query_proj.build([None, None, self.hidden_dim])
        self.key_proj.build([None, None, self.hidden_dim])
        self.value_proj.build([None, None, self.hidden_dim])
        self.out_proj.build([None, None, self.hidden_dim])
        self.built = True

    def _transpose_for_scores(self, tensor, batch_size):
        """
        Adapted from https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/bert/modeling_tf_bert.py#L252 # noqa: E501
        """
        # [batch_size, seq_len, all_head_dim] ->
        # [batch_size, seq_len, num_heads, head_dim]
        tensor = ops.reshape(
            tensor, (batch_size, -1, self.num_heads, self.head_dim)
        )
        # [batch_size, seq_len, num_heads, head_dim] ->
        # [batch_size, num_heads, seq_len, head_dim]
        return ops.transpose(tensor, axes=[0, 2, 1, 3])

    def call(
        self,
        x,
        attention_mask=None,
        return_attention_scores=None,
        training=False,
    ):
        batch_size = ops.shape(x)[0]
        mixed_query_layer = self.query_proj(inputs=x)
        mixed_key_layer = self.key_proj(inputs=x)
        mixed_value_layer = self.value_proj(inputs=x)
        query_layer = self._transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self._transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self._transpose_for_scores(mixed_value_layer, batch_size)

        # Scaled dot product between key and query = raw attention scores.
        attention_scores = ops.matmul(
            query_layer, ops.transpose(key_layer, axes=[0, 1, 3, 2])
        )
        dk = ops.cast(ops.sqrt(self.head_dim), dtype=attention_scores.dtype)
        attention_scores = ops.divide(
            attention_scores, dk
        )  # (batch_size, num_heads, seq_len_q, seq_len_k)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in the
            # call() function)
            attention_scores = ops.add(attention_scores, attention_mask)

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        dropout_attention_probs = self.dropout_layer(
            inputs=attention_probs, training=training
        )

        attn_output = ops.matmul(dropout_attention_probs, value_layer)
        attn_output = ops.transpose(attn_output, axes=[0, 2, 1, 3])

        # (batch_size, seq_len_q, hidden_dim)
        attn_output = ops.reshape(
            attn_output, (batch_size, -1, self.hidden_dim)
        )

        attn_output = self.out_proj(attn_output, training=training)
        if return_attention_scores:
            return (attn_output, attention_probs)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class CLIPLayer(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        intermediate_size,
        intermediate_activation="quick_gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.attn = CLIPAttention(
            self.hidden_dim,
            self.num_heads,
            name="multi_head_attention",
        )
        self.intermediate_activation = intermediate_activation
        self.layer_norm_1 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_1"
        )
        self.mlp_dense_1 = keras.layers.Dense(
            self.intermediate_size,
            name="c_fc",
        )
        self.mlp_dense_2 = keras.layers.Dense(
            self.hidden_dim,
            name="c_proj",
        )
        self.layer_norm_2 = keras.layers.LayerNormalization(
            epsilon=1e-5, name="layer_norm_2"
        )
        if self.intermediate_activation == "quick_gelu":
            self.activation = quick_gelu
        else:
            self.activation = keras.layers.Activation(
                self.intermediate_activation, name="activation"
            )

    def compute_attention(
        self, x, causal_attention_mask=None, attention_mask=None
    ):
        mask = None
        if causal_attention_mask is not None:
            mask = (
                ops.cast(causal_attention_mask, dtype=x.dtype)
                if causal_attention_mask is not None
                else None
            )
        if attention_mask is not None:
            attention_mask = (
                ops.cast(attention_mask, dtype=x.dtype)
                if attention_mask is not None
                else None
            )
            mask = ops.add(causal_attention_mask, attention_mask)

        return self.attn(
            x,
            attention_mask=mask,
        )[0]

    def build(self, input_shape):
        self.attn.build(None)
        self.layer_norm_1.build([None, None, self.hidden_dim])
        self.mlp_dense_1.build([None, None, self.hidden_dim])
        self.mlp_dense_2.build([None, None, self.intermediate_size])
        self.layer_norm_2.build([None, None, self.hidden_dim])
        self.built = True

    def call(self, x, causal_attention_mask=None, attention_mask=None):
        residual = x
        x = self.layer_norm_1(x)
        x = self.compute_attention(
            x,
            causal_attention_mask=causal_attention_mask,
            attention_mask=attention_mask,
        )
        x = x + residual
        residual = x
        x = self.mlp_dense_1(self.layer_norm_2(residual))
        x = self.activation(x)
        x = self.mlp_dense_2(x)
        x = residual + x
        return x

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "intermediate_activation": self.intermediate_activation,
            }
        )
        return config


class CLIPEncoder(keras.layers.Layer):
    def __init__(
        self,
        width,
        num_layers,
        num_heads,
        intermediate_size,
        intermediate_activation,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = width
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.resblocks = [
            CLIPLayer(
                self.width,
                self.num_heads,
                self.intermediate_size,
                self.intermediate_activation,
            )
            for _ in range(self.num_layers)
        ]

    def build(self, input_shape):
        for block in self.resblocks:
            block.build(input_shape)
        self.built = True

    def call(
        self,
        x,
        causal_attention_mask=None,
        attention_mask=None,
        intermediate_output=None,
    ):
        if intermediate_output is not None:
            if intermediate_output < 0:
                intermediate_output = self.num_layers + intermediate_output
        intermediate = None
        for i, block in enumerate(self.resblocks):
            if i == intermediate_output:
                x = block(
                    x,
                    causal_attention_mask=causal_attention_mask,
                    attention_mask=attention_mask,
                )
                intermediate = ops.copy(x)
        return x, intermediate

    def compute_output_shape(self, inputs_shape):

        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "width": self.width,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
            }
        )
        return config


class CLIPEmbeddings(keras.layers.Layer):
    def __init__(
        self, hidden_dim, vocabulary_size=49408, num_positions=77, **kwargs
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocabulary_size = vocabulary_size
        self.num_positions = num_positions
        self.token_embedding = keras.layers.Embedding(
            vocabulary_size,
            hidden_dim,
            name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            num_positions,
            hidden_dim,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.token_embedding.build(input_shape)
        self.position_embedding.build([1, self.num_positions])
        self.built = True

    def call(self, input_tokens):
        return (
            self.token_embedding(input_tokens)
            + self.position_embedding.weights[0]
        )


class SDTokenizer:
    def __init__(
        self,
        max_length=77,
        pad_with_end=True,
        tokenizer=None,
        has_start_token=True,
        pad_to_max_length=True,
        min_length=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        empty = self.tokenizer.encode("")["input_ids"]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.max_word_length = 8

    def tokenize_with_weights(self, text: str):
        """Tokenize the text, with weight values - presume 1.0 for all and
        ignore other features here. The details aren't relevant for a reference
        impl, and weights themselves has weak effect on SD3."""

        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0))
        to_tokenize = text.replace("\n", " ").split(" ")
        to_tokenize = [x for x in to_tokenize if x != ""]
        for word in to_tokenize:
            batch.extend(
                [
                    (t, 1)
                    for t in self.tokenizer.encode(word)["input_ids"][
                        self.tokens_start : -1
                    ]
                ]
            )
        batch.append((self.end_token, 1.0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0)] * (self.max_length - len(batch)))
        if self.min_length is not None and len(batch) < self.min_length:
            batch.extend([(pad_token, 1.0)] * (self.min_length - len(batch)))
        return [batch]


class SDXLClipGTokenizer(SDTokenizer):
    def __init__(self, tokenizer):
        super().__init__(pad_with_end=False, tokenizer=tokenizer)


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


class CLIPTokenizer:
    """This code is adapted from
    https://github.com/divamgupta/stable-diffusion-tensorflow.
    """

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

    def encode_text(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token]
                for bpe_token in self.bpe(token).split(" ")
            )
        return [self.start_of_text] + bpe_tokens + [self.end_of_text]

    def encode(self, text):
        if isinstance(text, list):
            print("is a list")
            out = list(map(self.encode_text, text))
            # Find the maximum sequence length
            max_length = max(len(seq) for seq in out)

            # Pad the input sequences with zeros
            input_ids = [seq + [49407] * (max_length - len(seq)) for seq in out]

            # Create attention masks
            attention_mask = [
                [1] * len(seq) + [0] * (max_length - len(seq)) for seq in out
            ]
        else:
            input_ids = self.encode_text(text)
            attention_mask = [1] * len(input_ids)

        output_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return output_dict

    def get_vocab(self):
        return {symbol: index for index, symbol in enumerate(self.vocab)}

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


class SD3Tokenizer:
    def __init__(self):
        clip_tokenizer = CLIPTokenizer()
        self.clip_l = SDTokenizer(tokenizer=clip_tokenizer)
        self.clip_g = SDXLClipGTokenizer(clip_tokenizer)
        # TODO uncomment when T5XXLTokenizer is added
        # self.t5xxl = T5XXLTokenizer()

    def tokenize_with_weights(self, text: str):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text)
        out["l"] = self.clip_l.tokenize_with_weights(text)
        # TODO uncomment when T5XXLTokenizer is added
        # out["t5xxl"] = self.t5xxl.tokenize_with_weights(text)
        return out


class CLIPTextModel_(keras.Model):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_size,
        intermediate_activation,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = intermediate_activation
        self.embeddings = CLIPEmbeddings(hidden_dim)
        self.encoder = CLIPEncoder(
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_size,
            intermediate_activation,
        )
        self.final_layer_norm = keras.layers.LayerNormalization(axis=-1)

    def build(self, input_shape):
        self.embeddings.build(input_shape)
        self.encoder.build(input_shape)
        self.final_layer_norm.build([None, None, self.hidden_dim])

    def call(
        self,
        input_tokens,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
    ):
        x = self.embeddings(input_tokens)
        # Compute causal mask
        causal_mask = ops.ones((ops.shape(x)[1], ops.shape(x)[1]))
        causal_mask = ops.triu(causal_mask)
        causal_mask = ops.cast(causal_mask, "float32")
        x, i = self.encoder(
            x,
            causal_attention_mask=causal_mask,
            intermediate_output=intermediate_output,
        )
        x = self.final_layer_norm(x)
        if i is not None and final_layer_norm_intermediate:
            i = self.final_layer_norm(i)

        indices = ops.expand_dims(
            ops.cast(ops.argmax(input_tokens, axis=-1), "int32"), axis=-1
        )
        pooled_output = ops.take_along_axis(x, indices[:, :, None], axis=1)
        pooled_output = ops.squeeze(pooled_output)

        return x, i, pooled_output


class CLIPTextModel(keras.Model):
    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_size,
        intermediate_activation,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.text_model = CLIPTextModel_(
            num_layers,
            hidden_dim,
            num_heads,
            intermediate_size,
            intermediate_activation,
        )
        self.text_projection = keras.layers.Dense(
            units=hidden_dim, use_bias=False
        )

    def build(self, input_shape):
        self.text_model.build(input_shape)
        self.text_projection.build([None, hidden_dim])

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding.weights[0]

    def set_input_embeddings(self, embeddings):
        self.text_model.embeddings.token_embedding.weights[0].assign(embeddings)

    def call(self, *args, **kwargs):
        x = self.text_model(*args, **kwargs)
        out = self.text_projection(x[2])
        return (x[0], x[1], out, x[2])


class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        tokens = list(map(lambda a: a[0], token_weight_pairs[0]))
        out, pooled = self([tokens])
        if pooled is not None:
            first_pooled = pooled[0:1]
        else:
            first_pooled = pooled
        output = [out[0:1]]
        return ops.concatenate(output, axis=-2), first_pooled


class SDClipModel(keras.Model, ClipTokenWeightEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_size,
        intermediate_activation="quick_gelu",
        max_length=77,
        layer="last",
        layer_idx=None,
        model_class=CLIPTextModel,
        special_tokens={"start": 49406, "end": 49407, "pad": 49407},
        layer_norm_hidden_state=True,
        return_projected_pooled=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert layer in self.LAYERS
        self.model_class = model_class
        self.transformer = model_class(
            num_layers,
            hidden_dim,
            num_heads,
            intermediate_size,
            intermediate_activation,
        )
        self.num_layers = num_layers
        self.max_length = max_length
        self.transformer.build((None, None))
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.logit_scale = keras.Variable(4.6055)
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.set_clip_options({"layer": layer_idx})

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get(
            "projected_pooled", self.return_projected_pooled
        )
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def call(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        tokens = ops.cast(tokens, "int64")
        outputs = self.transformer(
            tokens,
            intermediate_output=self.layer_idx,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
        )
        self.transformer.set_input_embeddings(backup_embeds)
        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]
        pooled_output = None
        if len(outputs) >= 3:
            if (
                not self.return_projected_pooled
                and len(outputs) >= 4
                and outputs[3] is not None
            ):
                pooled_output = ops.cast(outputs[3], "float32")
            elif outputs[2] is not None:
                pooled_output = ops.cast(outputs[2], "float32")
        return ops.cast(z, "float32"), pooled_output


class SDXLClipG(SDClipModel):
    """Wraps the CLIP-G model into the SD-CLIP-Model interface"""

    def __init__(
        self,
        num_layers,
        hidden_dim,
        num_heads,
        intermediate_size,
        intermediate_activation="gelu",
        layer="penultimate",
        layer_idx=None,
        **kwargs,
    ):
        if layer == "penultimate":
            layer = "hidden"
            layer_idx = -2
        super().__init__(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            intermediate_activation=intermediate_activation,
            layer=layer,
            layer_idx=layer_idx,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False,
        )
