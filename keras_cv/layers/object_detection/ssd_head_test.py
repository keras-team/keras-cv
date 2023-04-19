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

from keras_cv.layers.object_detection.ssd_head import SSDHead


class SSDHeadTest(tf.test.TestCase):
    def test_num_parameters(self):
        input1 = tf.keras.Input((80, 80, 256))
        input2 = tf.keras.Input((40, 40, 512))
        input3 = tf.keras.Input((20, 20, 1024))

        output = SSDHead(num_anchors=[15, 20, 25],
                         num_classes=10)([input1, input2, input3])

        model = tf.keras.models.Model(
            inputs=[input1, input2, input3], outputs=output
        )

        keras_params = sum(
            [tf.keras.backend.count_params(p) for p in model.trainable_weights]
        )
        # Taken from original implementation
        original_params = 5000520

        self.assertEqual(keras_params, original_params)

    def test_shape(self):
        inputs = [
            tf.random.uniform((3, 80, 80, 256)),
            tf.random.uniform((3, 40, 40, 512)),
            tf.random.uniform((3, 20, 20, 1024)),
        ]

        output = SSDHead(num_anchors=[15, 20, 25],
                         num_classes=10)(inputs)

        self.assertEqual(type(output), dict)

        # Taken from original implementation
        self.assertEqual(output.get("classification_results").shape,
                         [3, 138000, 10])
        self.assertEqual(output.get("bbox_regression_results").shape,
                         [3, 138000, 4])
