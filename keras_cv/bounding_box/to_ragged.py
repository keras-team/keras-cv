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

import keras_cv.bounding_box.validate_format as validate_format


def to_ragged(bounding_boxes, sentinel=-1, dtype=tf.float32):
    """converts a Dense padded bounding box `tf.Tensor` to a `tf.RaggedTensor`.

    Bounding boxes are ragged tensors in most use cases. Converting them to a dense
    tensor makes it easier to work with Tensorflow ecosystem.
    This function can be used to filter out the masked out bounding boxes by
    checking for padded sentinel value of the class_id axis of the bounding_boxes.

    Usage:
    ```python
    bounding_boxes = {
        "boxes": tf.constant([[2, 3, 4, 5], [0, 1, 2, 3]]),
        "num_classes": tf.constant([[-1, 1]]),
    }
    bounding_boxes = bounding_box.to_ragged(bounding_boxes)
    print(bounding_boxes)
    # {
    #     "boxes": [[0, 1, 2, 3]],
    #     "num_classes": [[1]]
    # }
    ```

    Args:
        bounding_boxes: a Tensor of bounding boxes.  May be batched, or unbatched.
        sentinel: The value indicating that a bounding box does not exist at the current
            index, and the corresponding box is padding.  Defaults to -1.
        dtype: the data type to use for the underlying Tensors.
    Returns:
        dictionary of `tf.RaggedTensor` or 'tf.Tensor' containing the filtered bounding
        boxes.
    """
    info = validate_format.validate_format(bounding_boxes)

    if info["ragged"]:
        return bounding_boxes

    boxes = bounding_boxes.get("boxes")
    num_classes = bounding_boxes.get("num_classes")
    mask = num_classes != sentinel

    boxes = tf.ragged.boolean_mask(boxes, mask)
    num_classes = tf.ragged.boolean_mask(num_classes, mask)

    if isinstance(boxes, tf.Tensor):
        boxes = tf.RaggedTensor.from_tensor(boxes)

    if isinstance(num_classes, tf.Tensor) and len(num_classes.shape) > 1:
        num_classes = tf.RaggedTensor.from_tensor(num_classes)

    result = bounding_boxes.copy()
    result["boxes"] = tf.cast(boxes, dtype)
    result["num_classes"] = tf.cast(num_classes, dtype)
    return result
