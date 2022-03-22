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

from keras_cv.utils import preprocessing


class Sharpen(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Performs the sharpen operation on given images.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.  Defaults to
            `[0, 255].`
        blend_factor: A float between 0 and 1 that controls that blending between the
            original image and the sharpened image. A value of `blend_factor=0.0` makes
            layer perform a no-op operation, while a value of 1.0 uses the sharpened
            result entirely.  Values between 0 and 1 result in linear interpolation
            between the original image and the sharpened image.

            Defaults to 1.0.
    """

    def __init__(
        self,
        value_range=(0, 255),
        blend_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.blend_factor = blend_factor

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(
            image, original_range=self.value_range, target_range=(0, 255)
        )
        orig_image = image
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, axis=0)

        # SMOOTH PIL Kernel.
        kernel = (
            tf.constant(
                [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
            )
            / 13.0
        )
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]

        degenerate = tf.nn.depthwise_conv2d(
            image, kernel, strides, padding="VALID", dilations=[1, 1]
        )
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(degenerate, axis=0)

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])

        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)
        # Blend the final result.
        result = preprocessing.blend(result, orig_image, self.blend_factor)
        result = preprocessing.transform_value_range(
            result, original_range=(0, 255), target_range=self.value_range
        )
        return result
