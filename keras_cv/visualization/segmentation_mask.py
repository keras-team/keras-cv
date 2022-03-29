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
"""Utility function to help visualize batch of binary mask.
"""

import tensorflow as tf

from keras_cv.visualization.colors import colors


def draw_segmentation(image, mask, color="red", alpha=0.4):
    """Draws segmentation masks on images with desired color
    and transparency.
    Colors supported are standard X11 colors.

    Args:
        image: an uint8 tensor with shape (N, img_height, img_height, 3)
        mask: an uint8 tensor of shape (N, img_height, img_height) with values
            between either 0 or 1.
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.4)
    Returns:
        Masks overlaid on images.

    Raises:
        ValueError: On incorrect data type for images or masks.
    """

    def _blend(image1, image2, factor):
        difference = image2 - image1
        scaled = factor * difference
        # Do addition in float.
        blended = image1 + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return tf.round(blended)
        # We need to clip and then cast.
        blended = tf.round(tf.clip_by_value(blended, 0.0, 255.0))
        return blended

    tf.debugging.assert_integer(
        image, message="Only integer dtypes supported for images."
    )
    tf.debugging.assert_integer(
        mask, message="Only integer dtypes supported for masks."
    )

    if tf.math.reduce_any(tf.math.logical_and(mask != 1, mask != 0)):
        raise ValueError("`mask` elements should be in [0, 1]")
    if image.shape[:3] != mask.shape:
        raise ValueError(
            f"image.shape[:3] == mask.shape should be true, got {image.shape[:3]} != {mask.shape}"
        )
    if alpha <= 0.0:
        return image

    _input_image_dtype = image.dtype

    # compute colored mask
    rgb = colors.get(color, None)
    if not rgb:
        raise ValueError(
            f"{color} is not supported yet,"
            "please check supported colors at `keras_cv.visualization.colors`"
        )
    solid_color = tf.expand_dims(tf.ones_like(mask), axis=-1)
    color = tf.cast(tf.reshape(list(rgb), [1, 1, 1, 3]), dtype=tf.uint8)
    solid_color *= color
    colored_mask = tf.expand_dims(mask, axis=-1) * solid_color

    # blend mask with image
    image = tf.cast(image, tf.float32)
    colored_mask = tf.cast(colored_mask, tf.float32)
    if alpha >= 1.0:
        masked_image = colored_mask
    else:
        masked_image = _blend(image, colored_mask, alpha)

    # stack masks along channel.
    mask_3d = tf.stack([mask] * 3, axis=-1)

    # exclude non positive area on image
    masked_image = tf.where(tf.cast(mask_3d, tf.bool), masked_image, image)
    masked_image = tf.cast(masked_image, dtype=_input_image_dtype)
    return masked_image
