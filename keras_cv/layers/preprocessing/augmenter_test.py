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

from keras_cv.layers import preprocessing


class AugmenterTest(tf.test.TestCase):
    def test_return_shapes(self):
        input = tf.ones((2, 512, 512, 3))

        layer = preprocessing.Augmenter(
            [
                preprocessing.Grayscale(
                    output_channels=1,
                ),
                preprocessing.RandomCropAndResize(
                    target_size=(100, 100),
                    crop_area_factor=(1, 1),
                    aspect_ratio_factor=(1, 1),
                ),
            ]
        )

        output = layer(input, training=True)

        self.assertEqual(output.shape, [2, 100, 100, 1])

    def test_in_tf_function(self):
        input = tf.ones((2, 512, 512, 3))

        layer = preprocessing.Augmenter(
            [
                preprocessing.Grayscale(
                    output_channels=1,
                ),
                preprocessing.RandomCropAndResize(
                    target_size=(100, 100),
                    crop_area_factor=(1, 1),
                    aspect_ratio_factor=(1, 1),
                ),
            ]
        )

        @tf.function
        def augment(x):
            return layer(x, training=True)

        output = augment(input)

        self.assertEqual(output.shape, [2, 100, 100, 1])
