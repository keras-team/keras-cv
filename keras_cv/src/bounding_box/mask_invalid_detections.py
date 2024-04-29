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

from keras_cv.src import backend
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.to_ragged import to_ragged
from keras_cv.src.bounding_box.validate_format import validate_format


@keras_cv_export("keras_cv.bounding_box.mask_invalid_detections")
def mask_invalid_detections(bounding_boxes, output_ragged=False):
    """masks out invalid detections with -1s.

    This utility is mainly used on the output of non-max suppression operations.
    The output of non-max-suppression contains all the detections, even invalid
    ones. Users are expected to use `num_detections` to determine how many boxes
    are in each image.

    In contrast, KerasCV expects all bounding boxes to be padded with -1s.
    This function uses the value of `num_detections` to mask out
    invalid boxes with -1s.

    Args:
        bounding_boxes: a dictionary complying with KerasCV bounding box format.
            In addition to the normal required keys, these boxes are also
            expected to have a `num_detections` key.
        output_ragged: whether to output RaggedTensor based bounding
            boxes.
    Returns:
        bounding boxes with proper masking of the boxes according to
        `num_detections`. This allows proper interop with non-max supression.
        Returned boxes match the specification fed to the function, so if the
        bounding box tensor uses `tf.RaggedTensor` to represent boxes the
        returned value will also return `tf.RaggedTensor` representations.
    """
    # ensure we are complying with KerasCV bounding box format.
    info = validate_format(bounding_boxes)
    if info["ragged"]:
        raise ValueError(
            "`bounding_box.mask_invalid_detections()` requires inputs to be "
            "Dense tensors. Please call "
            "`bounding_box.to_dense(bounding_boxes)` before passing your boxes "
            "to `bounding_box.mask_invalid_detections()`."
        )
    if "num_detections" not in bounding_boxes:
        raise ValueError(
            "`bounding_boxes` must have key 'num_detections' "
            "to be used with `bounding_box.mask_invalid_detections()`."
        )

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    confidence = bounding_boxes.get("confidence", None)
    num_detections = bounding_boxes.get("num_detections")

    # Create a mask to select only the first N boxes from each batch
    mask = ops.cast(
        ops.expand_dims(ops.arange(boxes.shape[1]), axis=0),
        num_detections.dtype,
    )
    mask = mask < num_detections[:, None]

    classes = ops.where(mask, classes, -ops.ones_like(classes))

    if confidence is not None:
        confidence = ops.where(mask, confidence, -ops.ones_like(confidence))

    # reuse mask for boxes
    mask = ops.expand_dims(mask, axis=-1)
    mask = ops.repeat(mask, repeats=boxes.shape[-1], axis=-1)
    boxes = ops.where(mask, boxes, -ops.ones_like(boxes))

    result = bounding_boxes.copy()

    result["boxes"] = boxes
    result["classes"] = classes
    if confidence is not None:
        result["confidence"] = confidence

    if output_ragged and backend.supports_ragged():
        return to_ragged(result)

    return result
