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

from keras_cv.models.object_detection_3d import CenterPillarBackbone
from keras_cv.tests.test_case import TestCase


class CenterPillarBackboneTest(TestCase):
    def test_output_shape(self):
        x = tf.random.normal((1, 16, 16, 5))
        model = CenterPillarBackbone(
            input_shape=(16, 16, 5),
            stackwise_down_blocks=[6, 2, 1],
            stackwise_down_filters=[128, 256, 512],
            stackwise_up_filters=[512, 256, 256],
        )
        output = model(x)
        self.assertEqual(output.shape, x.shape[:-1] + (256))

    def test_model_size(self):
        model = CenterPillarBackbone(
            input_shape=(16, 16, 5),
            stackwise_down_blocks=[6, 2, 1],
            stackwise_down_filters=[128, 256, 512],
            stackwise_up_filters=[512, 256, 256],
        )
        self.assertLen(model.layers, 125)

    def test_preset(self):
        model = CenterPillarBackbone.from_preset(
            "waymo_open_dataset", input_shape=(16, 16, 5)
        )
        x = tf.random.normal((1, 16, 16, 5))
        _ = model(x)
