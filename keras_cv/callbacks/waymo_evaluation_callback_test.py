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

import keras_cv
from keras_cv.callbacks import WaymoEvaluationCallback
from keras_cv.metrics.coco.pycoco_wrapper import METRIC_NAMES
from keras_cv.models.object_detection.__test_utils__ import _create_bounding_box_dataset

NUM_RECORDS = 10
POINT_FEATURES = 3
NUM_BOXES = 2
BOX_FEATURES = 8

METRIC_KEYS = [
    "average_precision",
    "average_precision_ha_weighted",
    "precision_recall",
    "precision_recall_ha_weighted",
    "breakdown",
]


class WaymoEvaluationCallbackTest(tf.test.TestCase):
    def test_model_fit(self):
        # Silly hypothetical model
        model = keras.Sequential(layers=[layers.Dense(BOX_FEATURES)])
        model.compile(optimizer="adam", loss="mse")

        fake_dataset = tf.data.Dataset.from_tensor_slices(
            (
                {
                    "point_clouds": tf.random.normal((NUM_RECORDS, POINT_FEATURES)),
                    "bounding_boxes": tf.random.normal(
                        NUM_RECORDS, NUM_BOXES, BOX_FEATURES
                    ),
                }
            )
        )

        callback = WaymoEvaluationCallback(validation_data=fake_dataset)
        history = model.fit(fake_dataset, callbacks=[callback])

        self.assertAllInSet(METRIC_KEYS, history.history.keys())
