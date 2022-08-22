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
"""Utility functions for working with bounding boxes."""

import tensorflow as tf

from keras_cv import bounding_box


def clip_to_image(bounding_boxes, images, bounding_box_format):
    """clips bounding boxes to image boundaries"""
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="rel_xyxy",
        images=images,
    )
    bounding_boxes, images, squeeze = _format_inputs(bounding_boxes, images)
    x1, y1, x2, y2, rest = tf.split(
        bounding_boxes, [1, 1, 1, 1, bounding_boxes.shape[-1] - 4], axis=-1
    )
    clipped_bounding_boxes = tf.concat(
        [
            tf.clip_by_value(x1, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(y1, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(x2, clip_value_min=0, clip_value_max=1),
            tf.clip_by_value(y2, clip_value_min=0, clip_value_max=1),
            rest,
        ],
        axis=-1,
    )

    clipped_bounding_boxes = bounding_box.convert_format(
        clipped_bounding_boxes,
        source="rel_xyxy",
        target=bounding_box_format,
        images=images,
    )
    clipped_bounding_boxes = _format_outputs(clipped_bounding_boxes, squeeze)
    return clipped_bounding_boxes


def _format_inputs(boxes, images):
    boxes_rank = len(boxes.shape)
    if boxes_rank > 3:
        raise ValueError(
            "Expected len(boxes.shape)=2, or len(boxes.shape)=3, got "
            f"len(boxes.shape)={boxes_rank}"
        )
    boxes_includes_batch = boxes_rank == 3
    # Determine if images needs an expand_dims() call
    if images is not None:
        images_rank = len(images.shape)
        if images_rank > 4:
            raise ValueError(
                "Expected len(images.shape)=2, or len(images.shape)=3, got "
                f"len(images.shape)={images_rank}"
            )
        images_include_batch = images_rank == 4
        if boxes_includes_batch != images_include_batch:
            raise ValueError(
                "clip_to_image() expects both boxes and images to be batched, or both "
                f"boxes and images to be unbatched.  Received len(boxes.shape)={boxes_rank}, "
                f"len(images.shape)={images_rank}.  Expected either len(boxes.shape)=2 AND "
                "len(images.shape)=3, or len(boxes.shape)=3 AND len(images.shape)=4."
            )
        if not images_include_batch:
            images = tf.expand_dims(images, axis=0)

    if not boxes_includes_batch:
        return tf.expand_dims(boxes, axis=0), images, True
    return boxes, images, False


def _format_outputs(boxes, squeeze):
    if squeeze:
        return tf.squeeze(boxes, axis=0)
    return boxes


def pad_with_sentinels(bounding_boxes, default_value=-1):
    """Pads the given bounding box tensor with -1s.

    This is done to convert RaggedTensors to be converted into
    standard Dense tensors, which have better performance and
    compatibility within the TensorFlow ecosystem.

    Args:
        bounding_boxes: a ragged tensor of bounding boxes.
        Can be batched or unbatched.

    Returns:
        a Tensor containing the -1 padded bounding boxes.
    """
    return bounding_boxes.to_tensor(default_value)


def filter_sentinels(bounding_boxes, default_value=-1):
    """converts a Dense padded bounding box `Tensor` to a `tf.RaggedTensor`.

    Args:
        bounding_boxes: a Tensor of bounding boxes.  May be batched, or unbatched.

    Returns:
        `tf.RaggedTensor`or 'tf.Tensor' containing the filtered bounding boxes.
    """
    is_ragged = isinstance(bounding_boxes, tf.RaggedTensor)
    if is_ragged:
        bounding_boxes = bounding_box.pad_with_sentinels(
            bounding_boxes, default_value=default_value
        )
    mask = bounding_boxes[..., 4] != default_value
    filtered_bounding_boxes = tf.ragged.boolean_mask(bounding_boxes, mask)
    return filtered_bounding_boxes


def add_class_id(bounding_boxes, class_id=0):
    """Add class ID to the innermost Tensor or RaggedTensor representing bounding boxes.

    Bounding box utilities in Keras_CV expects bounding boxes to have class IDs
    along with bounding box cordinates. This utility adds a class ID to the
    innermost tensor representing the bounding boxes.

    Usage:
    ```python
    bounding_boxes = tf.random.uniform(shape=[2, 2, 4])
    bounding_boxes_with_class_id = keras_cv.bounding_box.add_class_id(
                                    bounding_boxes, class_id=1)
    bounding_boxes_with_class_id is a Tensor of shape [2, 2, 5]
    ```

    Args:
        bounding_boxes: a `tf.Tensor` of bounding_boxes, may be batched unbatched.
        class_id: (Optional) The value of class id that needs to be padded.
            Defaults to 0.

    Returns:
        `tf.Tensor` with an additional class id padded to the original bounding boxes.
    """
    # format input bounding boxes
    is_ragged = isinstance(bounding_boxes, tf.RaggedTensor)

    if is_ragged:
        row_lengths = list(bounding_boxes.nested_row_lengths())
        # increase row length to account for clas-id addition
        row_lengths[1] = row_lengths[1] + 1
        bounding_boxes = bounding_boxes.to_tensor()

    # pad input bounding boxes
    if bounding_boxes.shape[-1] != 4:
        raise ValueError(
            "The number of values along the bounding box axis is "
            "expected to be 4. But got {}.".format(bounding_boxes.shape[-1])
        )
    bounding_box_rank = len(tf.shape(bounding_boxes))
    if bounding_box_rank == 2:
        paddings = tf.constant([[0, 0], [0, 1]])
    elif bounding_box_rank == 3:
        paddings = tf.constant([[0, 0], [0, 0], [0, 1]])
    else:
        raise ValueError(
            "The bounding boxes should be of rank 2 or 3. However "
            "add_class_id received bounding_boxes of rank {}.".format(bounding_box_rank)
        )

    bounding_boxes = tf.pad(
        bounding_boxes,
        paddings=paddings,
        mode="CONSTANT",
        constant_values=class_id,
    )

    # format output bounding boxes
    if is_ragged:
        bounding_boxes = tf.RaggedTensor.from_tensor(
            bounding_boxes,
            lengths=row_lengths,
        )
    return bounding_boxes
