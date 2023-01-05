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

import keras_cv.layers.maxvit_layers as maxvit_layers


class MaxViTLayersTest(tf.test.TestCase):
    def test_window_and_grid_wrong_sizes(self):
        with self.assertRaisesRegexp(
            ValueError,
            "window_size must not be a negative number. Received -1",
        ):
            maxvit_layers.WindowPartitioning(window_size=-1)

        with self.assertRaisesRegexp(
            ValueError,
            "grid_size must not be a negative number. Received -1",
        ):
            maxvit_layers.GridPartitioning(grid_size=-1)

    def test_indivisible_window_and_grid_input(self):
        input = tf.random.normal([1, 15, 15, 3])
        with self.assertRaisesRegexp(
            ValueError,
            "Feature map sizes (15, 15) not divisible by window size (4).",
        ):
            maxvit_layers.WindowPartitioning(window_size=4)(input)

        with self.assertRaisesRegexp(
            ValueError,
            "Feature map sizes (15, 15) not divisible by grid size (4).",
        ):
            maxvit_layers.GridPartitioning(grid=4)(input)



    def test_maxvit_stem_output_shape_and_type(self):
        layer = maxvit_layers.MaxViTStem()
        inputs = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=1)
        output = layer(inputs)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 112, 112, 64])

    def test_maxvit_block_output_shape_and_type_with_stride(self):
        layer = maxvit_layers.MaxViTBlock(
            hidden_size=64, head_size=32, window_size=7, grid_size=7, pool_stride=2
        )
        inputs = tf.random.uniform((1, 224, 224, 12), minval=0, maxval=1)
        output = layer(inputs)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 112, 112, 64])

    def test_maxvit_block_output_shape_and_type_without_stride(self):
        layer = maxvit_layers.MaxViTBlock(
            hidden_size=64, head_size=32, window_size=7, grid_size=7, pool_stride=1
        )
        inputs = tf.random.uniform((1, 224, 224, 12), minval=0, maxval=1)
        output = layer(inputs)

        self.assertTrue(isinstance(output, tf.Tensor))
        self.assertLen(output, 1)
        self.assertEquals(output.shape, [1, 224, 224, 64])
