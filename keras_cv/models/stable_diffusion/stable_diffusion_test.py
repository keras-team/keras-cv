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

import pytest
from tensorflow.keras import mixed_precision

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models import StableDiffusion
from keras_cv.tests.test_case import TestCase


@pytest.mark.extra_large
class StableDiffusionTest(TestCase):
    def test_end_to_end_golden_value(self):
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)

        img = stablediff.text_to_image(prompt, seed=1337, num_steps=5)
        self.assertAllClose(
            img[0][13:14, 13:14, :][0][0], [0, 135, 75], atol=1e-4
        )

        # Verify that the step-by-step creation flow creates an identical output
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(
            img,
            stablediff.generate_image(text_encoding, seed=1337, num_steps=5),
            atol=1e-4,
        )

    def test_image_encoder_golden_value(self):
        stablediff = StableDiffusion(128, 128)

        outputs = stablediff.image_encoder.predict(ops.ones((1, 128, 128, 3)))
        self.assertAllClose(
            outputs[0][1:4][0][0],
            [2.451568, 1.607522, -0.546311, -1.194388],
            atol=1e-4,
        )

    @pytest.mark.tf_keras_only
    def test_mixed_precision(self):
        mixed_precision.set_global_policy("mixed_float16")
        stablediff = StableDiffusion(128, 128)
        _ = stablediff.text_to_image("Testing123 haha!")

    def test_generate_image_rejects_noise_and_seed(self):
        stablediff = StableDiffusion(128, 128)

        with self.assertRaisesRegex(
            ValueError,
            r"`diffusion_noise` and `seed` should not both be passed",
        ):
            _ = stablediff.generate_image(
                stablediff.encode_text("thou shall not render"),
                diffusion_noise=keras.random.normal((1, 16, 16, 4), seed=42),
                seed=1337,
            )


if __name__ == "__main__":
    tf.test.main()
