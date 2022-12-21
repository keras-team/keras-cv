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
import keras
import pytest
import tensorflow as tf

try:
    from keras_cv.callbacks import WaymoEvaluationCallback
except ImportError:
    pass

NUM_RECORDS = 10
POINT_FEATURES = 3
NUM_POINTS = 20
NUM_BOXES = 2
BOX_FEATURES = 9

METRIC_KEYS = [
    "average_precision",
    "average_precision_ha_weighted",
    "precision_recall",
    "precision_recall_ha_weighted",
    "breakdown",
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
                    tf.fill((NUM_RECORDS // 2, NUM_BOXES, BOX_FEATURES), -1), tf.float32
                ),
            ],
            axis=0,
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                points,
                {
                    "boxes": boxes,
                },
            )
        ).batch(5)

        callback = WaymoEvaluationCallback(validation_data=dataset)
        history = model.fit(points, boxes, callbacks=[callback])

        self.assertAllInSet(METRIC_KEYS, history.history.keys())

    def build_model(self):
        inputs = tf.keras.Input(shape=(POINT_FEATURES, NUM_POINTS))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(BOX_FEATURES * NUM_BOXES)(x)
        x = keras.layers.Reshape((NUM_BOXES, BOX_FEATURES))(x)
        x = keras.layers.Lambda(lambda x: (x[:, :, :7], x[:, :, 7:]))(x)

        class MeanLoss(keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(y_pred, axis=-1)

        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(loss=MeanLoss())

        return model
