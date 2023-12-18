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

from keras_cv.backend import ops
from keras_cv.backend import random
from keras_cv.backend.keras import mixed_precision
from keras_cv.models import StableDiffusion
from keras_cv.tests.test_case import TestCase


@pytest.mark.tf_only
class StableDiffusionTest(TestCase):
    @pytest.mark.large
    def test_end_to_end_golden_value(self):
        self.skipTest("TODO: #2246 values differ for Keras2 and Keras3 TF")
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)

        img = stablediff.text_to_image(prompt, seed=1337, num_steps=5)
        self.assertAllEqual(img[0][13:14, 13:14, :][0][0], [66, 38, 185])

        # Verify that the step-by-step creation flow creates an identical output
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(
            img,
            stablediff.generate_image(text_encoding, seed=1337, num_steps=5),
            atol=1e-4,
        )

    @pytest.mark.extra_large
    def test_image_encoder_golden_value(self):
        stablediff = StableDiffusion(128, 128)

        outputs = stablediff.image_encoder.predict(ops.ones((1, 128, 128, 3)))
        self.assertAllClose(
            outputs[0][1:4][0][0],
            [2.451568, 1.607522, -0.546311, -1.194388],
            atol=5e-4,
        )

    @pytest.mark.extra_large
    def test_text_encoder_golden_value(self):
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(
            text_encoding[0][1][0:5],
            [0.029033, -1.325784, 0.308457, -0.061469, 0.03983],
            atol=1e-4,
        )

    @pytest.mark.extra_large
    def test_text_tokenizer_golden_value(self):
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)
        text_encoding = stablediff.tokenizer.encode(prompt)
        self.assertEqual(
            text_encoding[0:5],
            [49406, 320, 27111, 9038, 320],
        )

    @pytest.mark.extra_large
    def test_mixed_precision(self):
        try:
            mixed_precision.set_global_policy("mixed_float16")
            stablediff = StableDiffusion(128, 128)
            _ = stablediff.text_to_image("Testing123 haha!", num_steps=2)
        except Exception as e:
            raise (e)
        finally:
            # Clean up global policy
            mixed_precision.set_global_policy("float32")

    @pytest.mark.extra_large
    def test_generate_image_rejects_noise_and_seed(self):
        stablediff = StableDiffusion(128, 128)

        with self.assertRaisesRegex(
            ValueError,
            r"`diffusion_noise` and `seed` should not both be passed",
        ):
            _ = stablediff.generate_image(
                stablediff.encode_text("thou shall not render"),
                diffusion_noise=random.normal((1, 16, 16, 4), seed=42),
                seed=1337,
            )


@pytest.mark.extra_large
class StableDiffusionMultiFrameworkTest(TestCase):
    @pytest.mark.filterwarnings("ignore::UserWarning")  # Torch + jit_compile
    def test_end_to_end(self):
        prompt = "a caterpillar smoking a hookah while sitting on a mushroom"
        stablediff = StableDiffusion(128, 128)

        img = stablediff.text_to_image(prompt, seed=1337, num_steps=5)

        # Verify that the step-by-step creation flow creates an identical output
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(
            img,
            stablediff.generate_image(text_encoding, seed=1337, num_steps=5),
            atol=1e-4,
        )
