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

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models import ResNet50V2Backbone
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)
from keras_cv.models.object_detection.faster_rcnn import FasterRCNN
from keras_cv.utils.train import get_feature_extractor


class FasterRCNNTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        ((2, 640, 480, 3),),
        ((2, 512, 512, 3),),
        ((2, 224, 224, 3),),
    )
    def test_faster_rcnn_infer(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=self._build_backbone(),
        )
        images = tf.random.normal(batch_shape)
        outputs = model(images, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    @parameterized.parameters(
        ((2, 640, 480, 3),),
        ((2, 512, 512, 3),),
        ((2, 224, 224, 3),),
    )
    def test_faster_rcnn_train(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=self._build_backbone(),
        )
        images = tf.random.normal(batch_shape)
        outputs = model(images, training=True)
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    def test_invalid_compile(self):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="yxyx",
            backbone=self._build_backbone(),
        )
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(rpn_box_loss="binary_crossentropy")
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(
                rpn_classification_loss=keras.losses.BinaryCrossentropy(
                    from_logits=False
                )
            )

    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.  To run the test please run: \n"
        "`INTEGRATION=true pytest keras_cv/",
    )
    def test_faster_rcnn_with_dictionary_input_format(self):
        faster_rcnn = keras_cv.models.FasterRCNN(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=self._build_backbone(),
        )

        images, boxes = _create_bounding_box_dataset("xywh")
        dataset = tf.data.Dataset.from_tensor_slices(
            {"images": images, "bounding_boxes": boxes}
        ).batch(5, drop_remainder=True)

        faster_rcnn.compile(
            optimizer=optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

        faster_rcnn.fit(dataset, epochs=1)
        faster_rcnn.evaluate(dataset)

    def _build_backbone(self):
        backbone = ResNet50V2Backbone()
        extractor_levels = [2, 3, 4, 5]
        extractor_layer_names = [
            backbone.pyramid_level_inputs[i] for i in extractor_levels
        ]
        return get_feature_extractor(
            backbone, extractor_layer_names, extractor_levels
        )
