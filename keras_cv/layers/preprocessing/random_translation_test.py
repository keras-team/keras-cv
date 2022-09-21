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
from absl.testing import parameterized

from keras_cv.layers.preprocessing.random_translation import RandomTranslation


class RandomTranslationTest(tf.test.TestCase, parameterized.TestCase):
    def test_horizontal_translation(self):
        images = tf.cast(
            np.ones((4, 4, 3)),
            tf.float32,
        )
        images = tf.expand_dims(images, axis=0)
        bboxes = tf.cast(
            tf.constant([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            tf.float32,
        )
        bboxes = tf.expand_dims(bboxes, axis=0)
        layer = RandomTranslation(
            x_factor=[0.5, 0.5],
            bounding_box_format="rel_xyxy",
            fill_mode="constant",
            fill_value=0.0,
        )
        inp = {"images": images, "bounding_boxes": bboxes}
        outputs = layer(inp, training=True)
        output_images, output_bboxes = outputs["images"], outputs["bounding_boxes"]
        expected_bboxes = np.array([[[0.5, 0.0, 1.0, 1.0, 0.0]]])
        self.assertAllClose(expected_bboxes, output_bboxes)
        expected_images = np.ones((4, 4, 3)).astype(np.float32)
        expected_images[:, :2, :] = 0
        expected_images = np.expand_dims(expected_images, axis=0)
        self.assertAllClose(expected_images, output_images)

    def test_vertical_translation(self):
        images = tf.cast(
            np.ones((4, 4, 3)),
            tf.float32,
        )
        images = tf.expand_dims(images, axis=0)
        bboxes = tf.cast(
            tf.constant([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            tf.float32,
        )
        bboxes = tf.expand_dims(bboxes, axis=0)
        layer = RandomTranslation(
            y_factor=[0.5, 0.5],
            bounding_box_format="rel_xyxy",
            fill_mode="constant",
            fill_value=0.0,
        )
        inp = {"images": images, "bounding_boxes": bboxes}
        outputs = layer(inp, training=True)
        output_images, output_bboxes = outputs["images"], outputs["bounding_boxes"]
        expected_bboxes = np.array([[[0.0, 0.5, 1.0, 1.0, 0.0]]])
        self.assertAllClose(expected_bboxes, output_bboxes)
        expected_images = np.ones((4, 4, 3)).astype(np.float32)
        expected_images[:2, :, :] = 0
        expected_images = np.expand_dims(expected_images, axis=0)
        self.assertAllClose(expected_images, output_images)

    def test_vertical_and_horizontal_translation(self):
        images = tf.cast(
            np.ones((4, 4, 3)),
            tf.float32,
        )
        images = tf.expand_dims(images, axis=0)
        bboxes = tf.cast(
            tf.constant([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            tf.float32,
        )
        bboxes = tf.expand_dims(bboxes, axis=0)
        layer = RandomTranslation(
            x_factor=[0.25, 0.25],
            y_factor=[0.25, 0.25],
            bounding_box_format="rel_xyxy",
            fill_mode="constant",
            fill_value=0.0,
        )
        inp = {"images": images, "bounding_boxes": bboxes}
        outputs = layer(inp, training=True)
        output_images, output_bboxes = outputs["images"], outputs["bounding_boxes"]
        expected_bboxes = np.array([[[0.25, 0.25, 1.0, 1.0, 0.0]]])
        self.assertAllClose(expected_bboxes, output_bboxes)
        expected_images = np.ones((4, 4, 3)).astype(np.float32)
        expected_images[:1, :, :] = 0
        expected_images[:, :1, :] = 0
        expected_images = np.expand_dims(expected_images, axis=0)
        self.assertAllClose(expected_images, output_images)

    def test_no_augmentation(self):
        """test for no image and bbox augmentation when x_factor,y_factor is 0,0"""
        images = tf.cast(
            np.ones((4, 4, 3)),
            tf.float32,
        )
        images = tf.expand_dims(images, axis=0)
        bboxes = tf.cast(
            tf.constant([[0.0, 0.0, 1.0, 1.0, 0.0]]),
            tf.float32,
        )
        bboxes = tf.expand_dims(bboxes, axis=0)
        layer = RandomTranslation(
            x_factor=0,
            y_factor=0,
            bounding_box_format="rel_xyxy",
        )
        inp = {"images": images, "bounding_boxes": bboxes}
        outputs = layer(inp, training=True)
        output_images, output_bboxes = outputs["images"], outputs["bounding_boxes"]
        self.assertAllClose(bboxes, output_bboxes)
        self.assertAllClose(images, output_images)
