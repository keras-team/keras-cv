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
from tensorflow import keras

from keras_cv.models.object_detection.yolox.layers import YoloXPAFPN
from keras_cv.tests.test_case import TestCase


class YoloXLabelEncoderTest(TestCase):
    def test_num_parameters(self):
        input1 = keras.Input((80, 80, 256))
        input2 = keras.Input((40, 40, 512))
        input3 = keras.Input((20, 20, 1024))

        output = YoloXPAFPN()({3: input1, 4: input2, 5: input3})

        model = keras.models.Model(
            inputs=[input1, input2, input3], outputs=output
        )

        keras_params = sum(
            [keras.backend.count_params(p) for p in model.trainable_weights]
        )
        # taken from original implementation
        original_params = 19523072

        self.assertEqual(keras_params, original_params)

    def test_output_shape(self):
        inputs = {
            3: tf.random.uniform((3, 80, 80, 256)),
            4: tf.random.uniform((3, 40, 40, 512)),
            5: tf.random.uniform((3, 20, 20, 1024)),
        }

        output1, output2, output3 = YoloXPAFPN()(inputs)

        self.assertEqual(output1.shape, [3, 80, 80, 256])
        self.assertEqual(output2.shape, [3, 40, 40, 512])
        self.assertEqual(output3.shape, [3, 20, 20, 1024])
