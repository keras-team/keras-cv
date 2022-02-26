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


def _axis_mask(axis_lengths, offsets, mask_len):
    axis_mask = tf.sequence_mask(axis_lengths, mask_len)
    rev_lengths = tf.minimum(offsets + axis_lengths, mask_len)
    axis_mask = tf.reverse_sequence(axis_mask, rev_lengths, seq_axis=1)
    return axis_mask


def xywh_to_mask(xywh, mask_shape):
    width, height = mask_shape
    cx = xywh[:, 0]
    cy = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    x0 = cx - (w / 2)
    y0 = cy - (h / 2)

    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)
    x0 = tf.cast(x0, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    w_mask = _axis_mask(w, x0, width)
    h_mask = _axis_mask(h, y0, height)

    w_mask = tf.expand_dims(w_mask, axis=-2)
    h_mask = tf.expand_dims(h_mask, axis=-1)
    masks = tf.logical_and(w_mask, h_mask)

    return masks


def corners_to_mask(corners, mask_shape):
    width, height = mask_shape
    x0 = corners[:, 0]
    y0 = corners[:, 1]
    x1 = corners[:, 2]
    y1 = corners[:, 3]
    w = x1 - x0
    h = y1 - y0

    w = tf.cast(w, tf.int32)
    h = tf.cast(h, tf.int32)
    x0 = tf.cast(x0, tf.int32)
    y0 = tf.cast(y0, tf.int32)
    w_mask = _axis_mask(w, x0, width)
    h_mask = _axis_mask(h, y0, height)

    w_mask = tf.expand_dims(w_mask, axis=-2)
    h_mask = tf.expand_dims(h_mask, axis=-1)
    masks = tf.logical_and(w_mask, h_mask)

    return masks


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
    masks = tf.less_equal(x0_rep, range_col)
    masks = masks & tf.less_equal(y0_rep, range_row)
    masks = masks & tf.less(range_col, x1_rep)
    masks = masks & tf.less(range_row, y1_rep)

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
