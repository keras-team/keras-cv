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

from keras_cv.utils import bounding_box


def rectangle_masks(corners, mask_shape):
    """Computes masks of rectangles

    Args:
        corners: tensor of rectangle coordinates with shape (batch_size, 4) in
            corners format (x0, y0, x1, y1).
        mask_shape: a shape tuple as (width, height) indicating the output
            width and height of masks.

    Returns:
        boolean masks with shape (batch_size, width, height) where True values
            indicate positions within rectangle coordinates.
    """
    # add broadcasting axes
    corners = corners[..., tf.newaxis, tf.newaxis]

    # split coordinates
    x0 = corners[:, 0]
    y0 = corners[:, 1]
    x1 = corners[:, 2]
    y1 = corners[:, 3]

    # repeat height and width
    width, height = mask_shape
    x0_rep = tf.repeat(x0, height, axis=1)
    y0_rep = tf.repeat(y0, width, axis=2)
    x1_rep = tf.repeat(x1, height, axis=1)
    y1_rep = tf.repeat(y1, width, axis=2)

    # range grid
    batch_size = tf.shape(corners)[0]
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


def fill_rectangle(images, centers_x, centers_y, widths, heights, fill_values):
    """Fill rectangles with fill value into images.

    Args:
        images: Tensor of images to fill rectangles into.
        centers_x: Tensor of positions of the rectangle centers on the x-axis.
        centers_y: Tensor of positions of the rectangle centers on the y-axis.
        widths: Tensor of widths of the rectangles
        heights: Tensor of heights of the rectangles
        fill_values: Tensor with same shape as images to get rectangle fill from.
    Returns:
        images with filled rectangles.
    """
    images_shape = tf.shape(images)
    images_height = images_shape[1]
    images_width = images_shape[2]

    xywh = tf.stack([centers_x, centers_y, widths, heights], axis=1)
    xywh = tf.cast(xywh, tf.float32)
    corners = bounding_box.xywh_to_corners(xywh)

    mask_shape = (images_width, images_height)
    is_rectangle = rectangle_masks(corners, mask_shape)
    is_rectangle = tf.expand_dims(is_rectangle, -1)

    images = tf.where(is_rectangle, fill_values, images)
    return images
