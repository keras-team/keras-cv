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


import numpy as np
import tensorflow as tf

from keras_cv import layers as cv_layers
from keras_cv.models.object_detection.object_detection_base_model import (
    ObjectDetectionBaseModel,
)


class ObjectDetectionBaseModelTest(tf.test.TestCase):
    def test_raises_error_when_y_provided_with_dataset(self):
        x = tf.data.Dataset.from_tensor_slices(
            (tf.ones((8, 512, 512, 3)), tf.ones((8, 4, 5)))
        )
        y = tf.constant(8, 4, 5)

        model = ObjectDetectionBaseModel(
            bounding_box_format="xywh", label_encoder=_default_encoder()
        )
        with self.assertRaisesRegex(ValueError, "When `x` is a `tf.data.Dataset`,"):
            model.fit(x=x, y=y)

    def test_numpy_array(self):
        model = DummySubclass()
        model.compile()

        x = np.ones((8, 512, 512, 3))
        y = np.ones((8, 4, 5))
        model.fit(x, y, validation_data=(x, y))
        model.evaluate(np.ones((8, 512, 512, 3)), np.ones((8, 4, 5)))

        model.train_on_batch(x, y)
        model.test_on_batch(x, y)

    def test_validation_split(self):
        model = DummySubclass()
        model.compile()

        x = np.ones((8, 512, 512, 3))
        y = np.ones((8, 4, 5))
        model.fit(x, y, validation_split=0.2)
        model.evaluate(np.ones((8, 512, 512, 3)), np.ones((8, 4, 5)))

    def test_tf_dataset(self):
        model = DummySubclass()
        model.compile()
        my_ds = tf.data.Dataset.from_tensor_slices(
            (np.ones((8, 512, 512, 3)), np.ones((8, 4, 5)))
        )
        my_ds = my_ds.batch(8)
        model.fit(my_ds, validation_data=my_ds)
        model.evaluate(np.ones((8, 512, 512, 3)), np.ones((8, 4, 5)))

    def test_with_sample_weight(self):
        pass


class DummySubclass(ObjectDetectionBaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            bounding_box_format="xywh", label_encoder=_default_encoder(), **kwargs
        )

    def train_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = data
        return {"loss": 0}

    def test_step(self, data):
        x, y = data
        y_for_metrics, y_training_target = data
        return {"loss": 0}


def _default_encoder():
    strides = [2**i for i in range(3, 8)]
    scales = [2**x for x in [0, 1 / 3, 2 / 3]]
    sizes = [32.0, 64.0, 128.0, 256.0, 512.0]
    aspect_ratios = [0.5, 1.0, 2.0]

    anchor_generator = cv_layers.AnchorGenerator(
        bounding_box_format="xywh",
        sizes=sizes,
        aspect_ratios=aspect_ratios,
        scales=scales,
        strides=strides,
        clip_boxes=True,
    )
    return cv_layers.RetinaNetLabelEncoder(
        bounding_box_format="xywh", anchor_generator=anchor_generator
    )
