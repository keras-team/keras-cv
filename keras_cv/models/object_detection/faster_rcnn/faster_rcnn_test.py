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
from absl.testing import parameterized

from keras_cv.backend import keras
from keras_cv.models import ResNet18V2Backbone
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)
from keras_cv.models.object_detection.faster_rcnn.faster_rcnn import FasterRCNN
from keras_cv.tests.test_case import TestCase


class FasterRCNNTest(TestCase):
    # TODO(ianstenbit): Make FasterRCNN support shapes that are not multiples
    # of 128, perhaps by adding a flag to the anchor generator for whether to
    # include anchors centered outside of the image. (RetinaNet does use those,
    # while FasterRCNN doesn't). For more context on why this is the case, see
    # https://github.com/keras-team/keras-cv/pull/1882
    @parameterized.parameters(
        ((2, 640, 384, 3),),
        ((2, 512, 512, 3),),
        ((2, 128, 128, 3),),
    )
    def test_faster_rcnn_infer(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=ResNet18V2Backbone(),
        )
        images = tf.random.normal(batch_shape)
        outputs = model(images, training=False)
        # 1000 proposals in inference
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    @parameterized.parameters(
        ((2, 640, 384, 3),),
        ((2, 512, 512, 3),),
        ((2, 128, 128, 3),),
    )
    def test_faster_rcnn_train(self, batch_shape):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="xyxy",
            backbone=ResNet18V2Backbone(),
        )
        images = tf.random.normal(batch_shape)
        outputs = model(images, training=True)
        self.assertAllEqual([2, 1000, 81], outputs[1].shape)
        self.assertAllEqual([2, 1000, 4], outputs[0].shape)

    def test_invalid_compile(self):
        model = FasterRCNN(
            num_classes=80,
            bounding_box_format="yxyx",
            backbone=ResNet18V2Backbone(),
        )
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(rpn_box_loss="binary_crossentropy")
        with self.assertRaisesRegex(ValueError, "only accepts"):
            model.compile(
                rpn_classification_loss=keras.losses.BinaryCrossentropy(
                    from_logits=False
                )
            )

    @pytest.mark.large  # Fit is slow, so mark these large.
    def test_faster_rcnn_with_dictionary_input_format(self):
        faster_rcnn = FasterRCNN(
            num_classes=20,
            bounding_box_format="xywh",
            backbone=ResNet18V2Backbone(),
        )

        images, boxes = _create_bounding_box_dataset("xywh")
        dataset = tf.data.Dataset.from_tensor_slices(
            {"images": images, "bounding_boxes": boxes}
        ).batch(5, drop_remainder=True)

        faster_rcnn.compile(
            optimizer=keras.optimizers.Adam(),
            box_loss="Huber",
            classification_loss="SparseCategoricalCrossentropy",
            rpn_box_loss="Huber",
            rpn_classification_loss="BinaryCrossentropy",
        )

        faster_rcnn.fit(dataset, epochs=1)
        faster_rcnn.evaluate(dataset)
