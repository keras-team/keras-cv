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

from keras_cv.bounding_box.converters import convert_format

H_AXIS = -3
W_AXIS = -2


def transform_from_corners_fn(
    bounding_boxes,
    *,
    transform_corners_fn,
    bounding_box_format,
    images=None,
    clip_boxes=True,
):
    """Transforms bounding boxes by applying a function to each of its corners.

    Args:
      bounding_boxes: a 2D (unbatched) or 3D (batched) Tensor of
        bounding boxes, potentially ragged.
      transform_corners_fn: a function to apply to each of the corners
        of the bounding boxes. It should take a 3D (unbatched) 4D
        tensor (batched) with last dimension 2 and return a tensor of
        the same sized. The tensor may also be ragged if
        `bounding_box` is ragged. The format of the corner will be
        "xy".
      bounding_box_format: the input/output format of the bounding
        boxes.
      images: the tensor of images, in HWC or BHWC format the bounding
        boxes refers to.
      clip_boxes: if `True`, clips the resulting bounding boxes to the
        input image sizes.

    Returns:
      A Tensor (potentially ragged) of the same size than
        `bounding_boxes` which are the bounding box of the transformed
        corners of `bounding_box`.
    """

    if clip_boxes and images is None:
        raise ValueError(
            "images are required to clip bounding boxes to their size. "
            "`transform_from_corners_fn()` received "
            f"`clip_boxes`={clip_boxes} and images={images}"
        )

    bounding_boxes = convert_format(
        bounding_boxes, source=bounding_box_format, target="xyxy", images=images
    )

    l, u, r, b, rest = tf.split(
        bounding_boxes, [1, 1, 1, 1, tf.shape(bounding_boxes)[-1] - 4], axis=-1
    )

    corners = tf.concat(
        [
            tf.stack([l, u], axis=-1),
            tf.stack([r, u], axis=-1),
            tf.stack([r, b], axis=-1),
            tf.stack([l, b], axis=-1),
        ],
        axis=-2,
    )

    corners_out = transform_corners_fn(corners)

    min_cordinates = tf.math.reduce_min(corners_out, axis=-2)
    max_cordinates = tf.math.reduce_max(corners_out, axis=-2)

    if clip_boxes:
        images_shape = tf.cast(tf.shape(images), max_cordinates.dtype)[-2:-4:-1]
        min_cordinates = tf.clip_by_value(
            min_cordinates, clip_value_min=0.0, clip_value_max=images_shape
        )
        max_cordinates = tf.clip_by_value(
            max_cordinates, clip_value_min=0.0, clip_value_max=images_shape
        )

    bounding_boxes_out_xyxy = tf.concat([min_cordinates, max_cordinates], axis=-1)

    bounding_boxes_out = convert_format(
        bounding_boxes_out_xyxy,
        source="xyxy",
        target=bounding_box_format,
        images=images,
    )

    return tf.concat([bounding_boxes_out, rest], axis=-1)
