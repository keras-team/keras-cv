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
from packaging import version

from keras_cv.models.resnet_v2_backbone import ResNetV2Backbone


class ResNetV2BackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.input_batch = tf.ones(shape=(self.batch_size, 224, 224, 3))

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
        )
        model(self.input_batch)

    def test_valid_call_with_rescaling(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=True,
        )
        model(self.input_batch)

    def test_valid_call_pooling(self):
        model = ResNetV2Backbone(
            stackwise_filters=[64, 128, 256, 512],
            stackwise_blocks=[2, 2, 2, 2],
            stackwise_strides=[1, 2, 2, 2],
            include_rescaling=False,
            pooling="avg",
        )
        model(self.input_batch)
