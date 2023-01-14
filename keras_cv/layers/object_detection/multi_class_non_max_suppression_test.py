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

import unittest

import tensorflow as tf

from keras_cv import layers as cv_layers


def decode_predictions_output_shapes():
    classes = 10
    predictions_shape = (8, 98208, 4 + classes)

    predictions = tf.random.stateless_uniform(
        shape=predictions_shape, seed=(2, 3), minval=0.0, maxval=1.0, dtype=tf.float32
    )
    box_pred = predictions[..., :4]
    confidence_pred = predictions[..., 4:]

    layer = cv_layers.MultiClassNonMaxSuppression(
        bounding_box_format="xyxy",
        from_logits=True,
        max_detections=100,
    )

    result = layer(box_prediction=box_pred, confidence_prediction=confidence_pred)
    return result


class NmsPredictionDecoderTest(tf.test.TestCase):
    def test_decode_predictions_output_shapes(self):
        result = decode_predictions_output_shapes()
        self.assertEqual(result["boxes"].shape, [8, 100, 4])
        self.assertEqual(result["classes"].shape, [8, 100])
        self.assertEqual(result["confidence"].shape, [8, 100])


@unittest.expectedFailure
class NmsPredictionDecoderTestWithXLA(tf.test.TestCase):
    def test_decode_predictions_output_shapes(self):
        xla_function = tf.function(decode_predictions_output_shapes, jit_compile=True)
        result = xla_function()
        self.assertEqual(result["boxes"].shape, [8, 100, 4])
        self.assertEqual(result["classes"].shape, [8, 100])
        self.assertEqual(result["confidence"].shape, [8, 100])


class NmsPredictionDecoderTestWithXLAMlirBridge(tf.test.TestCase):
    def setUp(self):
        tf.config.experimental.enable_mlir_bridge()

    def tearDown(self):
        tf.config.experimental.disable_mlir_bridge()

    # @unittest.expectedFailure
    def test_decode_predictions_output_shapes(self):
        xla_function = tf.function(decode_predictions_output_shapes, jit_compile=True)
        result = xla_function()
        self.assertEqual(result["boxes"].shape, [8, 100, 4])
        self.assertEqual(result["classes"].shape, [8, 100])
        self.assertEqual(result["confidence"].shape, [8, 100])
