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

from keras_cv.bounding_box.to_dense import to_dense
from keras_cv.bounding_box.to_ragged import to_ragged
from keras_cv.bounding_box.validate_format import validate_format


def mask_invalid_detections(bounding_boxes):
    """masks out invalid detections with -1s.

    This utility is mainly used on the output of `tf.image.combined_non_max_suppression`
    operations.  The output of `tf.image.combined_non_max_suppression` contains padding
    values of 0 instead of -1.  KerasCV expects all bounding boxes to be padded with
    -1s instead of 0s.  This function uses the value of `num_detections` to mask out
    invalid boxes with -1s.

    Args:
        bounding_boxes: a dictionary complying with KerasCV bounding box format.  In
            addition to the normal required keys, these boxes are also expected to have
            a `num_detections` key.
    """
    # ensure we are complying with KerasCV bounding box format.
    info = validate_format(bounding_boxes)
    if "num_detections" not in bounding_boxes:
        raise ValueError(
            "`bounding_boxes` must have key 'num_detections' "
            "to be used with `bounding_box.mask_invalid_detections()`."
        )

    # required to use tf.where()
    bounding_boxes = to_dense(bounding_boxes)

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    num_detections = bounding_boxes.get("num_detections")

    # Create a mask to select only the first N boxes from each batch
    mask = tf.repeat(
        tf.expand_dims(tf.range(boxes.shape[1]), axis=0), repeats=boxes.shape[0], axis=0
    )
    mask = mask < num_detections[:, None]

    classes = tf.where(mask, classes, -tf.ones_like(classes))

    # resuse mask for boxes
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.repeat(mask, repeats=boxes.shape[-1], axis=-1)
    boxes = tf.where(mask, boxes, -tf.ones_like(boxes))

    result = bounding_boxes.copy()
    result["boxes"] = boxes
    result["classes"] = classes

    if info["ragged"]:
        return to_ragged(result)

    return result
