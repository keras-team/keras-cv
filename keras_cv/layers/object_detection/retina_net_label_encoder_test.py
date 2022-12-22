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


class RetinaNetLabelEncoderTest(tf.test.TestCase):
    def test_label_encoder_output_shapes(self):
        images_shape = (8, 512, 512, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10, 1)

        images = tf.random.uniform(shape=images_shape)
        boxes = tf.random.uniform(
            shape=boxes_shape, minval=0.0, maxval=1.0, dtype=tf.float32
        )
        classes = tf.random.uniform(
            shape=classes_shape, minval=0, maxval=5, dtype=tf.float32
        )
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
        encoder = cv_layers.RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xyxy",
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertEqual(box_targets.shape, [8, 49104, 4])
        self.assertEqual(class_targets.shape, [8, 49104])

    def test_all_negative_1(self):
        images_shape = (8, 512, 512, 3)
        boxes_shape = (8, 10, 5)

        images = tf.random.uniform(shape=images_shape)
        boxes = -tf.ones(shape=boxes_shape, dtype=tf.float32)
        classes = boxes[..., 4]
        boxes = boxes[..., :4]
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
        encoder = cv_layers.RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xyxy",
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertFalse(tf.math.reduce_any(tf.math.is_nan(box_targets)))
        self.assertFalse(tf.math.reduce_any(tf.math.is_nan(class_targets)))

    def test_ragged_encoding(self):
        images_shape = (2, 512, 512, 3)

        images = tf.random.uniform(shape=images_shape)
        box_targets = tf.ragged.stack(
            [
                tf.constant([[0, 0, 10, 10], [5, 5, 10, 10]], tf.float32),
                tf.constant([[0, 0, 10, 10]], tf.float32),
            ]
        )
        class_targets = tf.ragged.stack(
            [tf.constant([[1], [1]], tf.float32), tf.constant([[1]], tf.float32)]
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
        encoder = cv_layers.RetinaNetLabelEncoder(
            anchor_generator=anchor_generator,
            bounding_box_format="xywh",
        )

        box_targets, class_targets = encoder(images, box_targets, class_targets)

        # 49104 is the anchor generator shape
        self.assertEqual(box_targets.shape, [2, 49104, 4])
        self.assertEqual(class_targets.shape, [2, 49104])
