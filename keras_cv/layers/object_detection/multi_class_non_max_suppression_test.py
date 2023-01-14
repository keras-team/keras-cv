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

import pytest
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

        layer = cv_layers.MultiClassNonMaxSuppression(
            bounding_box_format="xyxy",
            from_logits=True,
            max_detections=100,
        )

        result = layer(box_prediction=box_pred, confidence_prediction=confidence_pred)

        self.assertEqual(result["boxes"].shape, [8, 100, 4])
        self.assertEqual(result["classes"].shape, [8, 100])
        self.assertEqual(result["confidence"].shape, [8, 100])

    @unittest.expectedFailure
    def test_with_xla(self):
        # TODO if the teardown is working correctly the next line is not required
        tf.config.experimental.disable_mlir_bridge()
        xla_function = tf.function(
            self.test_decode_predictions_output_shapes, jit_compile=True
        )
        xla_function()

    @pytest.fixture()
    def prepare_xla_mlir_bridge_compiled_function(self):
        tf.config.experimental.enable_mlir_bridge()
        xla_function = tf.function(
            self.test_decode_predictions_output_shapes, jit_compile=True
        )
        yield xla_function
        tf.config.experimental.disable_mlir_bridge()

    @pytest.fixture(autouse=True)
    def store_xla_mlir(self, prepare_xla_mlir_bridge_compiled_function):
        self._xla_function = prepare_xla_mlir_bridge_compiled_function

    def test_with_xla_mlir_bridge(self):
        self._xla_function()
