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

from keras_cv.utils import bbox


def rectangle_masks(mask_shape, corners):
    """Computes positional masks of rectangles in images

    Args:
        mask_shape: shape of the masks as [batch_size, height, width].
        x0: x-coordinates of
        y0:
        x1:
        y1:

    Returns:
        Boolean masks with True at rectangle positions.
    """
    # add broadcasting axes
    x0 = corners[:, 0, tf.newaxis, tf.newaxis]
    y0 = corners[:, 1, tf.newaxis, tf.newaxis]
    x1 = corners[:, 2, tf.newaxis, tf.newaxis]
    y1 = corners[:, 3, tf.newaxis, tf.newaxis]

    # repeat for height and width
    batch_size, height, width = mask_shape
    x0_rep = tf.repeat(x0, height, axis=1)
    y0_rep = tf.repeat(y0, width, axis=2)
    x1_rep = tf.repeat(x1, height, axis=1)
    y1_rep = tf.repeat(y1, width, axis=2)

    # range grid
    range_row = tf.range(0, height, dtype=corners.dtype)
    range_col = tf.range(0, width, dtype=corners.dtype)
    range_row = tf.repeat(range_row[tf.newaxis, :, tf.newaxis], batch_size, 0)
    range_col = tf.repeat(range_col[tf.newaxis, tf.newaxis, :], batch_size, 0)

    # boolean masks
    mask_x0 = tf.less_equal(x0_rep, range_col)
    mask_y0 = tf.less_equal(y0_rep, range_row)
    mask_x1 = tf.less(range_col, x1_rep)
    mask_y1 = tf.less(range_row, y1_rep)

    masks = mask_x0 & mask_y0 & mask_x1 & mask_y1

    return masks


def batch_fill_rectangle(images, center_x, center_y, width, height, fill):
    """Fill rectangles with fill value into images.

    Args:
        images: the images to fill rectangles into.
        center_x: positions of the rectangle centers on the x-axis.
        center_y: positions of the rectangle centers on the y-axis.
        width: widths of the rectangles
        height: heights of the rectangles
        fill: A tensor with same shape as image. Values at rectangle
         positions are used as fill.
    Returns:
        images with filled rectangles.
    """
    images_shape = tf.shape(images)
    batch_size = images_shape[0]
    images_height = images_shape[1]
    images_width = images_shape[2]

    xywh = tf.stack([center_x, center_y, width, height], axis=1)
    xywh = tf.cast(xywh, tf.float32)
    corners = bbox.xywh_to_corners(xywh)

    masks_shape = (batch_size, images_height, images_width)
    is_patch_mask = rectangle_masks(masks_shape, corners)
    is_patch_mask = tf.expand_dims(is_patch_mask, -1)

    images = tf.where(tf.equal(is_patch_mask, True), fill, images)
    return images


def fill_rectangle(image, center_x, center_y, width, height, fill=None):
    """Fill a rectangle in a given image using the value provided in replace.

    Args:
        image: the starting image to fill the rectangle on.
        center_x: the X center of the rectangle to fill
        center_y: the Y center of the rectangle to fill
        width: the width of the resulting rectangle
        height: the height of the resulting rectangle
        fill: A tensor with same shape as image. Values at rectangle
         position are used as fill.
    Returns:
        image: the modified image with the chosen rectangle filled.
    """
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    half_height = tf.cast(tf.math.ceil(height / 2), tf.int32)
    half_width = tf.cast(tf.math.ceil(width / 2), tf.int32)

    lower_pad = tf.maximum(0, center_y - half_height)
    upper_pad = tf.maximum(0, image_height - center_y - half_height)
    left_pad = tf.maximum(0, center_x - half_width)
    right_pad = tf.maximum(0, image_width - center_x - half_width)

    shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)

    image = tf.where(tf.equal(mask, 0), fill, image)
    return image
