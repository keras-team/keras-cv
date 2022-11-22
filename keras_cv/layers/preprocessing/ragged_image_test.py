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
from absl.testing import parameterized

from keras_cv import layers

CONSISTENT_OUTPUT_TEST_CONFIGURATIONS = []

DENSE_OUTPUT_TEST_CONFIGURATIONS = []

RAGGED_OUTPUT_TEST_CONFIGURATIONS = []


class RaggedImageTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters()
    def test_preserves_ragged_status(self, layer_cls, init_args):
        layer = layer_cls(**init_args)

    @parameterized.named_parameters()
    def test_converts_ragged_to_dense(self, layer_cls, init_args):
        layer = layer_cls(**init_args)

    @parameterized.named_parameters()
    def test_converts_dense_to_ragged(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
