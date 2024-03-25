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
import tensorflow as tf

from keras_cv.layers import preprocessing
from keras_cv.tests.test_case import TestCase


class GrayscaleTest(TestCase):
    def test_layer_basics(self):
        input_data = tf.ones((2, 52, 24, 3))
        init_kwargs = {
            "output_channels": 3,
        }
        self.run_preprocessing_layer_test(
            cls=preprocessing.Grayscale,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(2, 52, 24, 3),
        )

    @pytest.mark.tf_only
    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((10, 10, 3)), tf.ones((10, 10, 3))], axis=0),
            tf.float32,
        )

        # test 1
        layer = preprocessing.Grayscale(
            output_channels=1,
        )

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs1 = augment(xs)

        # test 2
        layer = preprocessing.Grayscale(
            output_channels=3,
        )

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs2 = augment(xs)

        self.assertEqual(xs1.shape, (2, 10, 10, 1))
        self.assertEqual(xs2.shape, (2, 10, 10, 3))
