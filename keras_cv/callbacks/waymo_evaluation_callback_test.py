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

import pytest
import tensorflow as tf
from tensorflow import keras

from keras_cv.callbacks import WaymoEvaluationCallback

NUM_RECORDS = 10
POINT_FEATURES = 3
NUM_POINTS = 20
NUM_BOXES = 2
BOX_FEATURES = 7

METRIC_KEYS = [
    "average_precision_vehicle_l1",
    "average_precision_vehicle_l2",
    "average_precision_ped_l1",
    "average_precision_ped_l2",
]


class WaymoEvaluationCallbackTest(tf.test.TestCase):
    @pytest.mark.skipif(True, reason="Requires Waymo Open Dataset")
    def test_model_fit(self):
        # Silly hypothetical model
        model = self.build_model()

        points = tf.random.normal((NUM_RECORDS, POINT_FEATURES, NUM_POINTS))
        # Some random boxes, and some -1 boxes (to mimic padding ragged boxes)
        boxes = tf.concat(
            [
                tf.random.uniform((NUM_RECORDS // 2, NUM_BOXES, BOX_FEATURES)),
                tf.cast(
                    tf.fill((NUM_RECORDS // 2, NUM_BOXES, BOX_FEATURES), -1),
                    tf.float32,
                ),
            ],
            axis=0,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                points,
                {
                    "3d_boxes": {
                        "boxes": boxes,
                        "classes": tf.ones((NUM_RECORDS, NUM_BOXES)),
                        "difficulty": tf.ones((NUM_RECORDS, NUM_BOXES)),
                        "mask": tf.concat(
                            [
                                tf.ones((NUM_RECORDS // 2, NUM_BOXES)),
                                tf.zeros((NUM_RECORDS // 2, NUM_BOXES)),
                            ],
                            axis=0,
                        ),
                    }
                },
            )
        ).batch(5)

        callback = WaymoEvaluationCallback(validation_data=dataset)
        history = model.fit(points, boxes, callbacks=[callback])

        self.assertAllInSet(METRIC_KEYS, history.history.keys())

    def build_model(self):
        inputs = keras.Input(shape=(POINT_FEATURES, NUM_POINTS))
        x = keras.layers.Flatten()(inputs)
        # Add extra features for class and confidence
        x = keras.layers.Dense(NUM_BOXES * (BOX_FEATURES + 2))(x)
        x = keras.layers.Reshape((NUM_BOXES, BOX_FEATURES + 2))(x)
        x = keras.layers.Lambda(
            lambda x: {
                "3d_boxes": {
                    "boxes": x[:, :, :7],
                    "classes": tf.abs(x[:, :, 7]),
                    "confidence": x[:, :, 8],
                }
            }
        )(x)

        class MeanLoss(keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(y_pred, axis=-1)

        model = keras.Model(inputs=inputs, outputs=x)
        model.compile(loss=MeanLoss())

        return model
