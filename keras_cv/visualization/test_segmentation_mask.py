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

from keras_cv.visualization import draw_segmentation, colors


class DrawSegmentationTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        IMG_SIZE = 6
        MASK_SIZE = 2
        COLOR_CODE1 = 1
        COLOR_CODE2 = 2

        self.color = "red"
        self.color_map = {COLOR_CODE1: "red", COLOR_CODE2: "green"}

        # create two different images,
        image1 = tf.ones((IMG_SIZE, IMG_SIZE, 3), tf.uint8) * 100
        image2 = tf.ones((IMG_SIZE, IMG_SIZE, 3), tf.uint8) * 200
        self.images = tf.stack([image1, image2], axis=0)

        # create two center rectangle masks.
        mask1 = tf.ones((MASK_SIZE, MASK_SIZE), tf.int32) * COLOR_CODE1
        mask2 = tf.ones((MASK_SIZE, MASK_SIZE), tf.int32) * COLOR_CODE2

        self.mask_y = IMG_SIZE - MASK_SIZE
        self.mask_pad = int((self.mask_y) / 2)
        paddings = tf.constant([[self.mask_pad] * 2] * 2)
        mask1 = tf.cast(tf.pad(mask1, paddings, "CONSTANT"), dtype=tf.uint8)
        mask2 = tf.cast(tf.pad(mask2, paddings, "CONSTANT"), dtype=tf.uint8)
        self.masks = tf.stack([mask1, mask2], axis=0)

        self.IMG_SIZE = IMG_SIZE
        self.MASK_SIZE = MASK_SIZE

    def test_draw_segmentation_base_case(self):
        images = draw_segmentation(self.images, self.masks, color="red")
        self.assertEqual(images.shape, self.images.shape)

    def test_draw_segmentation_full_factor(self):
        images = draw_segmentation(self.images, self.masks, color=self.color, alpha=1.0)
        mask_section = images[
            :, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y
        ]

        actual_images = tf.Variable(tf.identity(self.images))
        actual_images[
            :, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y
        ].assign(mask_section)

        self.assertAllEqual(images, actual_images)

    def test_draw_segmentation_no_factor(self):
        images = draw_segmentation(self.images, self.masks, color=self.color, alpha=0.0)
        self.assertAllEqual(self.images, images)

    def test_draw_segmentation_partial_factor(self):
        alpha = 0.5
        images = draw_segmentation(
            self.images, self.masks, color=self.color, alpha=alpha
        )
        color_rgb = colors.get(self.color)
        alpha_tf = tf.constant(alpha)

        image_section = tf.cast(
            self.images[:, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y],
            tf.float32,
        )
        actual_mask_section = tf.round(
            image_section * (1 - alpha_tf) + alpha_tf * color_rgb
        )
        actual_images = tf.Variable(tf.identity(self.images))
        actual_images[
            :, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y
        ].assign(tf.cast(actual_mask_section, actual_images.dtype))
        self.assertAllEqual(images, actual_images)

    def test_draw_segmentation_color_map_base_case(self):
        images = draw_segmentation(
            self.images, self.masks, color=self.color_map, alpha=1.0
        )
        mask_section = images[
            :, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y
        ]
        actual_images = tf.Variable(tf.identity(self.images))
        actual_images[
            :, self.MASK_SIZE : self.mask_y, self.MASK_SIZE : self.mask_y
        ].assign(mask_section)

        _color_masks = []
        for c in self.color_map.values():
            color_rgb = colors.get(c)
            _color_masks.append(tf.ones_like(mask_section[0]) * color_rgb)
        actual_mask_section = tf.stack(_color_masks, axis=0)
        self.assertAllEqual(images, actual_images)

    def test_draw_segmentation_exception_handling(self):

        # test color type handling.
        with self.assertRaisesRegex(TypeError, "Dict or string is excepted"):
            draw_segmentation(self.images, self.masks, color=-1)
        # test color distinct code.
        _missing_color_map = {-1: "red"}

        with self.assertRaisesRegex(
            TypeError, f"Color mapping {_missing_color_map} does not map completely"
        ):
            draw_segmentation(self.images, self.masks, color=_missing_color_map)

        # test mask and image shape
        with self.assertRaisesRegex(
            ValueError, "image.shape[:3] == mask.shape should be true"
        ):
            draw_segmentation(self.images, tf.constant(1, tf.uint8))

        # test mask and image dtypes.
        with self.assertRaisesRegex(TypeError, "Only integer dtypes supported"):
            draw_segmentation(tf.cast(self.images, tf.float32), self.masks)

        with self.assertRaisesRegex(TypeError, "Only integer dtypes supported"):
            draw_segmentation(self.images, tf.cast(self.masks, tf.float32))
