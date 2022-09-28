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

import tensorflow as tf
from tensorflow.keras import mixed_precision

from keras_cv.models import StableDiffusion


class StableDiffusionTest(tf.test.TestCase):
    def DISABLED_test_end_to_end_golden_value(self):
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)

        img = stablediff.text_to_image(prompt, seed=1337)
        self.assertAllClose(img[0][13:14, 13:14, :][0][0], [15, 248, 229], atol=1e-4)

        # Verify that the step-by-step creation flow creates an identical output
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(
            img, stablediff.generate_image(text_encoding, seed=1337), atol=1e-4
        )

    def DISABLED_test_mixed_precision(self):
        mixed_precision.set_global_policy("mixed_float16")
        stablediff = StableDiffusion(128, 128)
        _ = stablediff.text_to_image("Testing123 haha!")

    def DISABLED_test_add_tokens(self):
        stablediff = StableDiffusion(128, 128)
        old_token_weights = (
            stablediff.text_encoder.embedding.token_embedding.get_weights()
        )
        old_position_weights = (
            stablediff.text_encoder.embedding.position_embedding.get_weights()
        )
        self.assertEqual(len(old_token_weights), 1)
        self.assertEqual(len(old_position_weights), 1)

        stablediff.add_tokens("<my-silly-cat-token>")
        new_token_weights = (
            stablediff.text_encoder.embedding.token_embedding.get_weights()
        )

        self.assertEqual(len(new_token_weights), 1)
        self.assertAllEqual(old_token_weights[0][:49408], new_token_weights[0][:49408])
        self.assertEqual(len(new_token_weights[0]), 49409)

        stablediff.add_tokens("<keras-token>")
        new_token_weights = (
            stablediff.text_encoder.embedding.token_embedding.get_weights()
        )
        self.assertEqual(len(new_token_weights[0]), 49410)


if __name__ == "__main__":
    tf.test.main()
