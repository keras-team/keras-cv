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
from absl.testing import parameterized

import keras_cv


IMG_SIZE = 6
MASK_SIZE = 2
N = 2
COLOR_CODE1 = 1
COLOR_CODE2 = 2

colors_test_data = (
    ("basic_string", "red"),
    ("basic_tuple", (0, 255, 0)),
    ("color_string_map", {1: (0, 255, 0)}),
    ("color_tuple_map", {1: "red"}),
)

dtypes_test_data = (
    ("float32", tf.float32),
    ("uint8", tf.uint8),
    ("int16", tf.int16),
    ("int32", tf.int32),
)

color_map_test_data = (
    ("complete-map-str", {COLOR_CODE1: "red", COLOR_CODE2: "green"}),
    ("missing-map-str", {COLOR_CODE1: "green"}),
    ("complete-map-tuple", {COLOR_CODE1: (255, 0, 0), COLOR_CODE2: (0, 255, 0)}),
)


class DrawSegmentationTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()

        self.images = tf.zeros((N, IMG_SIZE, IMG_SIZE, 3), tf.float32)

        # create center rectangle mask.
        mask1 = tf.ones((MASK_SIZE, MASK_SIZE), tf.int32) * COLOR_CODE1
        mask2 = tf.ones((MASK_SIZE, MASK_SIZE), tf.int32) * COLOR_CODE2

        self.mask_y = IMG_SIZE - MASK_SIZE
        self.mask_pad = int((self.mask_y) / 2)
        paddings = tf.constant([[self.mask_pad] * 2] * 2)
        mask1 = tf.cast(tf.pad(mask1, paddings, "CONSTANT"), dtype=tf.uint8)
        mask2 = tf.cast(tf.pad(mask2, paddings, "CONSTANT"), dtype=tf.uint8)
        self.masks = tf.stack([mask1, mask2], axis=0)

    @parameterized.named_parameters(*colors_test_data)
    def test_draw_segmentation_base_case(self, color):
        images = keras_cv.visualization.draw_segmentation(
            self.images, self.masks, color=color
        )
        self.assertEqual(images.shape, self.images.shape)

    @parameterized.named_parameters(*dtypes_test_data)
    def test_draw_segmentation_dtypes(self, dtype):
        images = keras_cv.visualization.draw_segmentation(
            tf.cast(self.images, dtype), self.masks
        )
        self.assertEqual(images.shape, self.images.shape)

    @parameterized.named_parameters(
        ("full_factor", 1.0), ("partial_factor", 0.5), ("no_factor", 0.0)
    )
    def test_draw_segmentation_partial_factor(self, alpha):
        images = keras_cv.visualization.draw_segmentation(
            self.images, self.masks, alpha=alpha
        )
        color_rgb = colors.get("red")
        alpha_tf = tf.constant(alpha)

        image_section = tf.cast(
            self.images[:, MASK_SIZE : self.mask_y, MASK_SIZE : self.mask_y], tf.float32
        )
        actual_mask_section = tf.round(
            image_section * (1 - alpha_tf) + alpha_tf * color_rgb
        )
        actual_images = tf.Variable(tf.identity(self.images))
        actual_images[:, MASK_SIZE : self.mask_y, MASK_SIZE : self.mask_y].assign(
            tf.cast(actual_mask_section, actual_images.dtype)
        )
        self.assertAllEqual(images, actual_images)

    @parameterized.named_parameters(*color_map_test_data)
    def test_draw_segmentation_color_map_base_case(self, color):
        images = keras_cv.visualization.draw_segmentation(
            self.images, self.masks, color=color, alpha=1.0
        )
        mask_section = images[:, MASK_SIZE : self.mask_y, MASK_SIZE : self.mask_y]
        actual_images = tf.Variable(tf.identity(self.images))
        actual_images[:, MASK_SIZE : self.mask_y, MASK_SIZE : self.mask_y].assign(
            mask_section
        )

        _color_masks = []
        _all_color_codes = [COLOR_CODE1, COLOR_CODE2]
        for c in _all_color_codes:
            if c not in color.keys():
                color.update({c: "red"})
        for k, c in color.items():
            if isinstance(c, str):
                color_rgb = colors.get(c)
            else:
                color_rgb = c
            _color_masks.append(tf.ones_like(mask_section[0]) * color_rgb)
        actual_mask_section = tf.stack(_color_masks, axis=0)
        self.assertAllEqual(images, actual_images)

    def test_draw_segmentation_3d_image(self):
        images = keras_cv.visualization.draw_segmentation(
            self.images[0], self.masks[0], color=[255, 0, 0]
        )
        self.assertEqual(images.shape, self.images[0].shape)

    def test_draw_segmentation_exception_handling(self):

        # test color type handling.
        with self.assertRaisesRegex(
            TypeError, f"Want type(color)=dict or string, got type(color)"
        ):
            keras_cv.visualization.draw_segmentation(self.images, self.masks, color=-1)

        with self.assertRaisesRegex(
            ValueError, "image.shape[:3] == mask.shape should be true"
        ):
            keras_cv.visualization.draw_segmentation(
                self.images, tf.constant(1, tf.uint8)
            )

        with self.assertRaisesRegex(TypeError, "Only integer dtypes supported"):
            keras_cv.visualization.draw_segmentation(
                self.images, tf.cast(self.masks, tf.float32)
            )
