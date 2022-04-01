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
from collections import defaultdict

import tensorflow as tf

from keras_cv.visualization.colors import colors


def _blend(image1, image2, factor=0.4):
    difference = image2 - image1
    scaled = factor * difference
    # Do addition in float.
    blended = image1 + scaled

    # We need to clip and then cast.
    blended = tf.round(tf.clip_by_value(blended, 0.0, 255.0))
    return blended


def _map_color_on_mask(masks, color):

    # check distinct mask color codes with color mapping.
    distinct_mask_code = tf.unique(tf.reshape(masks, -1)).y
    color_rgb = defaultdict(tuple)

    if isinstance(color, str):
        for idx in distinct_mask_code:
            try:
                color_rgb.update({int(idx): colors[color]})
            except KeyError:
                raise KeyError(
                    f"{color} is not supported yet,"
                    "please check supported colors at `keras_cv.visualization.colors`"
                )

        color_keys = list(distinct_mask_code.numpy())

    else:
        if any([code not in distinct_mask_code for code in color.keys()]):
            raise ValueError(
                f"Color mapping {color} does not map completely\
                  with distint color codes present in masks: {distinct_mask_code}"
            )

        # map color code with RGB
        for code, name in color.items():
            try:
                color_rgb.update({code: colors[name]})
            except KeyError:
                raise KeyError(
                    f"{name} is not supported yet,"
                    "please check supported colors at `keras_cv.visualization.colors`"
                )
        color_keys = list(color.keys())

    keys_tensor = tf.constant(color_keys, dtype=tf.int32)
    colored_masks = tf.TensorArray(
        tf.int32, size=0, dynamic_size=True, clear_after_read=True
    )

    for c in range(3):
        vals_tensor = tf.constant([color_rgb[color][c] for color in color_keys])
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
            default_value=0,
        )
        colored_masks.write(c, table[tf.cast(masks, tf.int32)]).mark_used()

    return tf.transpose(colored_masks.stack(), list(range(1, masks.ndim + 1)) + [0])


def draw_segmentation(image, mask, color=None, alpha=0.4):
    """Draws segmentation masks on images with desired color
    and transparency.
    Colors supported are standard X11 colors.

    Args:
        image: An integer tensor with shape (N, img_height, img_height, 3)
        mask: An integer tensor of shape (N, img_height, img_height) with masks
              values corresponding to `color_map`.
        color: The color or colors to draw the segmentation map in.
               This can either be a single color, or a dictionary mapping from class IDs
               to colors.  Supported color formats are RGB tuples and strings. A full
               list of color strings is available in `keras_cv/visualization/colors.py`.
        alpha: Transparency value between 0 and 1. (default: 0.4)
    Returns:
        Masks overlaid on images.

    Raises:
        ValueError: On incorrect data type and shapes for images or masks.
        KeyError: On incorrect color string in `color_map`, or a class ID missing from
            the color ma.
    """

    if not isinstance(color, (dict, str)):
        raise TypeError(
            f"Want type(color)=dict or string, got type(color)={type(color)}."
        )

    if image.shape[:3] != mask.shape:
        raise ValueError(
            "image.shape[:3] == mask.shape should be true, got "
            f"{image.shape[:3]} != {mask.shape}"
        )
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha should be in the range [0, 1], got alpha={alpha}")
    _input_image_dtype = image.dtype

    # compute colored mask
    colored_mask = _map_color_on_mask(mask, color)

    # blend mask with image
    image = tf.cast(image, tf.float32)
    colored_mask = tf.cast(colored_mask, tf.float32)
    masked_image = _blend(image, colored_mask, alpha)

    # stack masks along channel.
    mask_3d = tf.stack([mask] * 3, axis=-1)

    # exclude non positive area on image
    masked_image = tf.where(tf.cast(mask_3d, tf.bool), masked_image, image)
    masked_image = tf.cast(masked_image, dtype=_input_image_dtype)
    return masked_image
