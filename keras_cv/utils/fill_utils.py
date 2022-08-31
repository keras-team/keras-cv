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

from keras_cv import bounding_box


def _axis_mask(starts, ends, mask_len):
    # index range of axis
    batch_size = tf.shape(starts)[0]
    axis_indices = tf.range(mask_len, dtype=starts.dtype)
    axis_indices = tf.expand_dims(axis_indices, 0)
    axis_indices = tf.tile(axis_indices, [batch_size, 1])

    # mask of index bounds
    axis_mask = tf.greater_equal(axis_indices, starts) & tf.less(axis_indices, ends)
    return axis_mask


def corners_to_mask(bounding_boxes, mask_shape):
    """Converts bounding boxes in corners format to boolean masks

    Args:
        bounding_boxes: tensor of rectangle coordinates with shape (batch_size, 4) in
            corners format (x0, y0, x1, y1).
        mask_shape: a shape tuple as (width, height) indicating the output
            width and height of masks.

    Returns:
        boolean masks with shape (batch_size, width, height) where True values
            indicate positions within bounding box coordinates.
    """
    mask_width, mask_height = mask_shape
    x0, y0, x1, y1 = tf.split(bounding_boxes, [1, 1, 1, 1], axis=-1)

    w_mask = _axis_mask(x0, x1, mask_width)
    h_mask = _axis_mask(y0, y1, mask_height)

    w_mask = tf.expand_dims(w_mask, axis=1)
    h_mask = tf.expand_dims(h_mask, axis=2)
    masks = tf.logical_and(w_mask, h_mask)
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
    corners = bounding_box.convert_format(xywh, source="center_xywh", target="xyxy")
    mask_shape = (images_width, images_height)
    is_rectangle = corners_to_mask(corners, mask_shape)
    is_rectangle = tf.expand_dims(is_rectangle, -1)

    images = tf.where(is_rectangle, fill_values, images)
    return images
