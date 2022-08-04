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
import tensorflow as tf

from keras_cv.layers.preprocessing.posterization import Posterization


class PosterizationTest(tf.test.TestCase):
    rng = tf.random.Generator.from_non_deterministic_state()

    def test_raises_error_on_invalid_bits_parameter(self):
        invalid_values = [-1, 0, 9, 24]
        for value in invalid_values:
            with self.assertRaises(ValueError):
                Posterization(bits=value, value_range=[0, 1])

    def test_raises_error_on_invalid_value_range(self):
        invalid_ranges = [(1,), [1, 2, 3]]
        for value_range in invalid_ranges:
            with self.assertRaises(ValueError):
                Posterization(bits=1, value_range=value_range)

    def test_single_image(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(224, 224, 3), maxval=256)
        expected_output = self._calc_expected_output(dummy_input, bits=bits)

        layer = Posterization(bits=bits, value_range=[0, 255])
        output = layer(dummy_input)

        self.assertAllEqual(output, expected_output)

    def _get_random_bits(self):
        return int(self.rng.uniform(shape=(), minval=1, maxval=9, dtype=tf.int32))

    def test_single_image_rescaled(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(224, 224, 3), maxval=1.0)
        expected_output = self._calc_expected_output(dummy_input * 255, bits=bits) / 255

        layer = Posterization(bits=bits, value_range=[0, 1])
        output = layer(dummy_input)

        self.assertAllClose(output, expected_output)

    def test_batched_input(self):
        bits = self._get_random_bits()
        dummy_input = self.rng.uniform(shape=(2, 224, 224, 3), maxval=256)

        expected_output = []
        for image in dummy_input:
            expected_output.append(self._calc_expected_output(image, bits=bits))
        expected_output = tf.stack(expected_output)

        layer = Posterization(bits=bits, value_range=[0, 255])
        output = layer(dummy_input)

        self.assertAllEqual(output, expected_output)

    def test_works_with_xla(self):
        dummy_input = self.rng.uniform(shape=(2, 224, 224, 3))
        layer = Posterization(bits=4, value_range=[0, 1])

        @tf.function(jit_compile=True)
        def apply(x):
            return layer(x)

        apply(dummy_input)

    @staticmethod
    def _calc_expected_output(image, bits):
        """Posterization in numpy, based on Albumentations:

        The algorithm is basically:
        1. create a lookup table of all possible input pixel values to pixel values
            after posterize
        2. map each pixel in the input to created lookup table.

        Source:
            https://github.com/albumentations-team/albumentations/blob/89a675cbfb2b76f6be90e7049cd5211cb08169a5/albumentations/augmentations/functional.py#L407
        """
        dtype = image.dtype
        image = tf.cast(image, tf.uint8)

        lookup_table = np.arange(0, 256, dtype=np.uint8)
        mask = ~np.uint8(2 ** (8 - bits) - 1)
        lookup_table &= mask

        return tf.cast(lookup_table[image], dtype)
