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


class StableDiffusioNTest(tf.test.TestCase):
    def DISABLED_test_end_to_end_golden_value(self):
        stablediff = StableDiffusion(128, 128)
        img = stablediff.text_to_image(
            "a caterpillar smoking a hookah while sitting on a mushroom", seed=123
        )
        self.assertAllClose(img[0][64:65, 64:65, :][0][0], [255, 232, 18], atol=1e-4)

    def DISABLED_test_mixed_precision(self):
        mixed_precision.set_global_policy("mixed_float16")
        stablediff = StableDiffusion(128, 128)
        _ = stablediff.text_to_image("Testing123 haha!")


if __name__ == "__main__":
    tf.test.main()
