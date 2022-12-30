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

from keras_cv import layers as cv_layers


class NmsPredictionDecoderTest(tf.test.TestCase):
    def test_decode_predictions_output_shapes(self):
        classes = 10
        predictions_shape = (8, 98208, 4 + classes)

        predictions = tf.random.uniform(
            shape=predictions_shape, minval=0.0, maxval=1.0, dtype=tf.float32
        )
        box_pred = predictions[..., :4]
        confidence_pred = predictions[..., 4:]

        layer = cv_layers.NmsDecoder(
            bounding_box_format="xyxy",
            from_logits=True,
            max_detections=100,
        )

        result = layer(box_pred=box_pred, confidence_pred=confidence_pred)

        self.assertEqual(result["boxes"].shape, [8, 100, 4])
        self.assertEqual(result["classes"].shape, [8, 100])
        self.assertEqual(result["confidence"].shape, [8, 100])
