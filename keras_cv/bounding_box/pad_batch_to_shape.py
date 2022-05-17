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


def pad_batch_to_shape(bounding_boxes, target_shape, padding_values=-1):
    """Pads a list of bounding boxes with -1s.

    Boxes represented by all -1s are ignored by COCO metrics.

    Sample usage:
    bounding_box = [[1, 2, 3, 4], [5, 6, 7, 8]]   # 2 bounding_boxes with with xywh or
        corners format.
    target_shape = [3, 4]   # Add 1 more dummy bounding_box
    result = pad_batch_to_shape(bounding_box, target_shape)
    # result == [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -1, -1, -1]]

    target_shape = [2, 5]   # Add 1 more index after the current 4 coordinates.
    result = pad_batch_to_shape(bounding_box, target_shape)
    # result == [[1, 2, 3, 4, -1], [5, 6, 7, 8, -1]]

    Args:
        bounding_boxes: tf.Tensor of bounding boxes in any format.
        target_shape: Target shape to pad bounding box to. This should have the same
            rank as the bounding_boxes. Note that if the target_shape contains any
            dimension that is smaller than the bounding box shape, then no value will be
            padded.
        padding_values: value to pad, defaults to -1 to mask out in coco metrics.
    Returns:
        bounding_boxes padded to target shape.

    Raises:
        ValueError, when target shape has smaller rank or dimension value when
            comparing with shape of bounding boxes.
    """
    bounding_box_shape = tf.shape(bounding_boxes)
    if len(bounding_box_shape) != len(target_shape):
        raise ValueError(
            "Target shape should have same rank as the bounding box. "
            f"Got bounding_box shape = {bounding_box_shape}, "
            f"target_shape = {target_shape}"
        )
    for dim in range(len(target_shape)):
        if bounding_box_shape[dim] > target_shape[dim]:
            raise ValueError(
                "Target shape should be larger than bounding box shape "
                "in all dimensions. "
                f"Got bounding_box shape = {bounding_box_shape}, "
                f"target_shape = {target_shape}"
            )
    paddings = [
        [0, target_shape[dim] - bounding_box_shape[dim]]
        for dim in range(len(target_shape))
    ]
    return tf.pad(
        bounding_boxes, paddings, mode="CONSTANT", constant_values=padding_values
    )
