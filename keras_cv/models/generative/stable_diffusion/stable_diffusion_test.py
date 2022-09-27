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

        # Using TF global random seed to guarantee that subsequent text-to-image
        # runs are seeded identically.
        tf.random.set_seed(8675309)
        img = stablediff.text_to_image(prompt)
        self.assertAllClose(img[0][64:65, 64:65, :][0][0], [124, 188, 114], atol=1e-4)

        # Verify that the step-by-step creation flow creates an identical output
        tf.random.set_seed(8675309)
        text_encoding = stablediff.encode_text(prompt)
        self.assertAllClose(img, stablediff.generate_image(text_encoding), atol=1e-4)

    def DISABLED_test_mixed_precision(self):
        mixed_precision.set_global_policy("mixed_float16")
        stablediff = StableDiffusion(128, 128)
        _ = stablediff.text_to_image("Testing123 haha!")

    def DISABLED_test_generate_image_rejects_noise_and_seed(self):
        stablediff = StableDiffusion(128, 128)

        with self.assertRaisesRegex(
            ValueError, r"`diffusion_noise` and `seed` should not both be passed"
        ):
            _ = stablediff.generate_image(
                stablediff.encode_text("thou shall not render"),
                diffusion_noise=tf.random.normal((1, 16, 16, 4)),
                seed=1337,
            )


if __name__ == "__main__":
    tf.test.main()
