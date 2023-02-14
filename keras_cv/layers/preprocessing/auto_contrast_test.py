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

from keras_cv.layers import BaseImageAugmentationLayer
from keras_cv.layers import preprocessing
from keras_cv.utils import preprocessing as utils_preprocessing


class OldAutoContrast(BaseImageAugmentationLayer):
    """Performs the AutoContrast operation on an image.

    Auto contrast stretches the values of an image across the entire available
    `value_range`.  This makes differences between pixels more obvious.  An example of
    this is if an image only has values `[0, 1]` out of the range `[0, 255]`, auto
    contrast will change the `1` values to be `255`.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
    """

    def __init__(
        self,
        value_range,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_image(self, image, transformation=None, **kwargs):
        original_image = image
        image = utils_preprocessing.transform_value_range(
            image,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )

        low = tf.reduce_min(tf.reduce_min(image, axis=0), axis=0)
        high = tf.reduce_max(tf.reduce_max(image, axis=0), axis=0)
        scale = 255.0 / (high - low)
        offset = -low * scale

        image = image * scale[None, None] + offset[None, None]
        result = tf.clip_by_value(image, 0.0, 255.0)
        result = utils_preprocessing.transform_value_range(
            result,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
        )
        # don't process NaN channels
        result = tf.where(tf.math.is_nan(result), original_image, result)
        return result

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(self, segmentation_mask, transformation, **kwargs):
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range})
        return config


class AutoContrastTest(tf.test.TestCase):
    def test_constant_channels_dont_get_nanned(self):
        img = tf.constant([1, 1], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))

    def test_auto_contrast_expands_value_range(self):
        img = tf.constant([0, 128], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 255.0))

    def test_auto_contrast_different_values_per_channel(self):
        img = tf.constant(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.float32
        )
        img = tf.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0, ..., 0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0, ..., 1] == 0.0))

        self.assertTrue(tf.math.reduce_any(ys[0, ..., 0] == 255.0))
        self.assertTrue(tf.math.reduce_any(ys[0, ..., 1] == 255.0))

        self.assertAllClose(
            ys,
            [
                [
                    [[0.0, 0.0, 0.0], [85.0, 85.0, 85.0]],
                    [[170.0, 170.0, 170.0], [255.0, 255.0, 255.0]],
                ]
            ],
        )

    def test_auto_contrast_expands_value_range_uint8(self):
        img = tf.constant([0, 128], dtype=tf.uint8)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 255.0))

    def test_auto_contrast_properly_converts_value_range(self):
        img = tf.constant([0, 0.5], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = preprocessing.AutoContrast(value_range=(0, 1))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))

    def test_is_consistent_with_non_vectorized_implementation(self):
        images = tf.random.uniform((16, 32, 32, 3))

        old_output = OldAutoContrast(value_range=(0, 1))(images)
        output = preprocessing.AutoContrast(value_range=(0, 1))(images)

        self.assertAllClose(old_output, output)
