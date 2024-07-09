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


import numpy as np

from keras_cv.models.stable_diffusion.v3.clip_utils import CLIPEmbeddings
from keras_cv.models.stable_diffusion.v3.clip_utils import CLIPEncoder
from keras_cv.models.stable_diffusion.v3.clip_utils import CLIPTokenizer
from keras_cv.models.stable_diffusion.v3.clip_utils import SD3Tokenizer
from keras_cv.tests.test_case import TestCase


class ClipUtilsTest(TestCase):
    def test_clip_encoder(self):
        # clip encoder calls CLIPAttention and CLIPLayer, both of which gets
        # verified with this test
        batch_size = 32
        sequence_length = 20
        hidden_dim = 64
        num_layers = 4
        heads = 8
        intermediate_size = 256
        clip_encoder = CLIPEncoder(
            hidden_dim, num_layers, heads, intermediate_size
        )
        dummy_input = np.random.rand(batch_size, sequence_length, hidden_dim)
        output = clip_encoder(dummy_input)
        self.assertEqual(output.shape, (32, 20, 64))

    def test_clip_embeddings(self):
        vocab_size = 100  # arbitrary vocab size
        num_positions = 10  # arbitrary number of positions
        hidden_dim = 32  # arbitrary embedding dimension
        input_tokens = np.random.randint(0, vocab_size, (5, 10))
        # Instantiate the CLIPEmbeddings layer
        embeddings_layer = CLIPEmbeddings(
            hidden_dim,
            vocab_size,
            num_positions,
        )

        output = embeddings_layer(input_tokens)
        self.assertEqual(output.shape, [5, 10, 32])

    def test_clip_tokenizer(self):
        # Sample texts for tokenization
        sample_texts = "a cat"
        # Tokenize the sample texts
        tokenizer_keras = CLIPTokenizer()
        output_dict = tokenizer_keras.encode(sample_texts)
        # Test with single text prompt
        expected_output_dict = {
            "input_ids": [49406, 320, 2368, 49407],
            "attention_mask": [1, 1, 1, 1],
        }
        self.assertEqual(output_dict, expected_output_dict)
        # Test get_vocab()
        vocab = tokenizer_keras.get_vocab()
        self.assertEqual(vocab["*"], 9)
        # Test with list of text prompts
        sample_texts = ["A cat", "A dog", "A computer"]
        output_dict = tokenizer_keras.encode(sample_texts)
        expected_output_dict = {
            "input_ids": [
                [49406, 320, 2368, 49407, 49407],
                [49406, 320, 1929, 49407, 49407],
                [49406, 320, 11639, 652, 49407],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ],
        }
        self.assertEqual(output_dict, expected_output_dict)

    def test_sd3_tokenizer(self):
        sd3_tokenizer = SD3Tokenizer()
        out_keras = sd3_tokenizer.tokenize_with_weights("A cat")
        expected_g_tokens = [(49406, 1.0), (320, 1), (2368, 1), (49407, 1.0)]
        expected_l_tokens = [(49406, 1.0), (320, 1), (2368, 1), (49407, 1.0)]
        expected_t5xxl_tokens = [(71, 1), (1712, 1), (1, 1.0), (0, 1.0)]
        self.assertEqual(out_keras["g"][0][:4], expected_g_tokens)
        self.assertEqual(out_keras["l"][0][:4], expected_l_tokens)
        # TODO - uncomment to test T5XXLTokenizer after it is added
        # self.assertEqual(out_keras["t5xxl"][0][:4], expected_t5xxl_tokens)