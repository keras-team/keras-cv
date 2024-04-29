# Copyright 2023 The KerasCV Authors
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
from absl.testing import parameterized

from keras_cv.src.backend import ops


class TestCase(tf.test.TestCase, parameterized.TestCase):
    """Base test case class for KerasCV. (Copied from KerasNLP)."""

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)

    def assertAllEqual(self, x1, x2, msg=None):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllEqual(x1, x2, msg=msg)

    def assertAllGreaterEqual(self, x1, x2):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllGreaterEqual(x1, x2)

    def assertAllLessEqual(self, x1, x2):
        x1 = tf.nest.map_structure(convert_to_numpy, x1)
        x2 = tf.nest.map_structure(convert_to_numpy, x2)
        super().assertAllLessEqual(x1, x2)


def convert_to_numpy(x):
    if ops.is_tensor(x) and not isinstance(x, tf.RaggedTensor):
        return ops.convert_to_numpy(x)
    return x
