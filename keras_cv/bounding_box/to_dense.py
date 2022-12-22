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

from keras_cv.bounding_box import validate


def _box_shape(batched, boxes_shape, max_boxes):
    # ensure we dont drop the final axis in RaggedTensor mode
    if max_boxes is None:
        shape = list(boxes_shape)
        shape[-1] = 4
        return shape
    if batched:
        return [None, max_boxes, 4]
    return [max_boxes, 4]


def _classes_shape(batched, classes_shape, max_boxes):
    if max_boxes is None:
        return None
    if batched:
        return [None, max_boxes] + classes_shape[2:]
    return [max_boxes] + classes_shape[2:]


def to_dense(bounding_boxes, max_boxes=None):
    """to_dense converts bounding boxes to Dense tensors

    Args:
        bounding_boxes: bounding boxes in KerasCV dictionary format.
        max_boxes: the maximum number of boxes, used to pad tensors to a given
            shape.  This can be used to make object detection pipelines TPU
            compatible.
    """
    info = validate.validate(bounding_boxes)

    # Already running in masked mode
    if not info["ragged"]:
        return bounding_boxes

    if isinstance(bounding_boxes["classes"], tf.RaggedTensor):
        bounding_boxes["classes"] = bounding_boxes["classes"].to_tensor(
            -1,
            shape=_classes_shape(
                info["is_batched"], bounding_boxes["classes"].shape, max_boxes
            ),
        )

    if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
        bounding_boxes["boxes"] = bounding_boxes["boxes"].to_tensor(
            -1,
            shape=_box_shape(
                info["is_batched"], bounding_boxes["boxes"].shape, max_boxes
            ),
        )

    return bounding_boxes
