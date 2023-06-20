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
import pytest
import tensorflow as tf

from keras_cv import backend
from keras_cv import layers as cv_layers
from keras_cv.backend import ops
from keras_cv.models.object_detection.retinanet import RetinaNetLabelEncoder


class RetinaNetLabelEncoderTest(tf.test.TestCase):
    def test_label_encoder_output_shapes(self):
        images_shape = (8, 512, 512, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = np.random.uniform(size=boxes_shape, low=0.0, high=1.0)
        classes = np.random.uniform(size=classes_shape, low=0, high=5)
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        encoder = RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xyxy",
        )
        bounding_boxes = {"boxes": boxes, "classes": classes}
        box_targets, class_targets = encoder(images, bounding_boxes)

        self.assertEqual(box_targets.shape, (8, 49104, 4))
        self.assertEqual(class_targets.shape, (8, 49104))

    def test_all_negative_1(self):
        images_shape = (8, 512, 512, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = -np.ones(shape=boxes_shape, dtype="float32")
        classes = -np.ones(shape=classes_shape, dtype="float32")

        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="yxyx",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        encoder = RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xyxy",
        )

        bounding_boxes = {"boxes": boxes, "classes": classes}
        box_targets, class_targets = encoder(images, bounding_boxes)

        self.assertFalse(ops.any(ops.isnan(box_targets)))
        self.assertFalse(ops.any(ops.isnan(class_targets)))

    @pytest.mark.skipif(
        backend.supports_ragged() is False,
        reason="Only TensorFlow supports raggeds",
    )
    def test_ragged_encoding(self):
        images_shape = (2, 512, 512, 3)

        images = tf.random.uniform(shape=images_shape)
        boxes = tf.ragged.stack(
            [
                tf.constant([[0, 0, 10, 10], [5, 5, 10, 10]], "float32"),
                tf.constant([[0, 0, 10, 10]], "float32"),
            ]
        )
        classes = tf.ragged.stack(
            [
                tf.constant([[1], [1]], "float32"),
                tf.constant([[1]], "float32"),
            ]
        )
        strides = [2**i for i in range(3, 8)]
        scales = [2**x for x in [0, 1 / 3, 2 / 3]]
        sizes = [x**2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        aspect_ratios = [0.5, 1.0, 2.0]

        anchor_generator = cv_layers.AnchorGenerator(
            bounding_box_format="xywh",
            sizes=sizes,
            aspect_ratios=aspect_ratios,
            scales=scales,
            strides=strides,
        )
        encoder = RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xywh",
        )

        bounding_boxes = {"boxes": boxes, "classes": classes}
        box_targets, class_targets = encoder(images, bounding_boxes)

        # 49104 is the anchor generator shape
        self.assertEqual(box_targets.shape, (2, 49104, 4))
        self.assertEqual(class_targets.shape, (2, 49104))
