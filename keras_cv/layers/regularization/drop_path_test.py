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

from keras_cv.layers.regularization.drop_path import DropPath

class DropPathTest(tf.test.TestCase):
	FEATURE_SHAPE = (1, 14, 14, 256)

	def test_input_unchanged_in_eval_mode(self):
		layer = DropPath(drop_rate=0.5)
		inputs = tf.random.uniform(self.FEATURE_SHAPE)

		outputs = layer(inputs, training=False)

		self.assertAllClose(inputs, outputs)


	def test_input_unchanged_with_rate_equal_to_zero(self):
		layer = DropPath(drop_rate=0)
		inputs = tf.random.uniform(self.FEATURE_SHAPE)

		outputs = layer(inputs, training=True)

		self.assertAllClose(inputs, outputs)

	def test_input_gets_partially_zeroed_out_in_train_mode(self):
		layer = DropPath(drop_rate=0.2)
		inputs = tf.random.uniform(self.FEATURE_SHAPE)

		outputs = layer(inputs, training=True)

		non_zeros_inputs = tf.math.count_nonzero(inputs, dtype=tf.int32)
		non_zeros_outputs = tf.math.count_nonzero(outputs, dtype=tf.int32)

		self.assertGreaterEqual(non_zeros_inputs, non_zeros_outputs)