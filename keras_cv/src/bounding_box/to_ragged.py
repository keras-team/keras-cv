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

import keras_cv.src.bounding_box.validate_format as validate_format
from keras_cv.src import backend
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export("keras_cv.bounding_box.to_ragged")
def to_ragged(bounding_boxes, sentinel=-1, dtype=tf.float32):
    """converts a Dense padded bounding box `tf.Tensor` to a `tf.RaggedTensor`.

    Bounding boxes are ragged tensors in most use cases. Converting them to a
    dense tensor makes it easier to work with Tensorflow ecosystem.
    This function can be used to filter out the masked out bounding boxes by
    checking for padded sentinel value of the class_id axis of the
    bounding_boxes.

    Example:
    ```python
    bounding_boxes = {
        "boxes": tf.constant([[2, 3, 4, 5], [0, 1, 2, 3]]),
        "classes": tf.constant([[-1, 1]]),
    }
    bounding_boxes = bounding_box.to_ragged(bounding_boxes)
    print(bounding_boxes)
    # {
    #     "boxes": [[0, 1, 2, 3]],
    #     "classes": [[1]]
    # }
    ```

    Args:
        bounding_boxes: a Tensor of bounding boxes. May be batched, or
            unbatched.
        sentinel: The value indicating that a bounding box does not exist at the
            current index, and the corresponding box is padding, defaults to -1.
        dtype: the data type to use for the underlying Tensors.
    Returns:
        dictionary of `tf.RaggedTensor` or 'tf.Tensor' containing the filtered
        bounding boxes.
    """
    if backend.supports_ragged() is False:
        raise NotImplementedError(
            "`bounding_box.to_ragged` was called using a backend which does "
            "not support ragged tensors. "
            f"Current backend: {keras.backend.backend()}."
        )

    info = validate_format.validate_format(bounding_boxes)

    if info["ragged"]:
        return bounding_boxes

    boxes = bounding_boxes.get("boxes")
    classes = bounding_boxes.get("classes")
    confidence = bounding_boxes.get("confidence", None)

    mask = classes != sentinel

    boxes = tf.ragged.boolean_mask(boxes, mask)
    classes = tf.ragged.boolean_mask(classes, mask)
    if confidence is not None:
        confidence = tf.ragged.boolean_mask(confidence, mask)

    if isinstance(boxes, tf.Tensor):
        boxes = tf.RaggedTensor.from_tensor(boxes)

    if isinstance(classes, tf.Tensor) and len(classes.shape) > 1:
        classes = tf.RaggedTensor.from_tensor(classes)

    if confidence is not None:
        if isinstance(confidence, tf.Tensor) and len(confidence.shape) > 1:
            confidence = tf.RaggedTensor.from_tensor(confidence)

    result = bounding_boxes.copy()
    result["boxes"] = tf.cast(boxes, dtype)
    result["classes"] = tf.cast(classes, dtype)

    if confidence is not None:
        result["confidence"] = tf.cast(confidence, dtype)

    return result
