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


def pad_with_sentinels(bounding_boxes):
    """Pads the given bounding box tensor with -1s. 

       This is done to convert RaggedTensors to be converted into
       standard Dense tensors, which have better performance and
       compatibility within the TensorFlow ecosystem.

    Args:
      bounding_boxes: a ragged tensor of bounding boxes.
      Can be batched or unbatched.

    Returns:
      bounding_boxes_tensor: a tensor containing the -1 padded bounding boxes.
    """
    return bounding_boxes.to_tensor(-1)


def filter_sentinels(bounding_boxes):
    """converts a Dense padded bounding box `Tensor` to a `tf.RaggedTensor`.

    Args:
      bounding_boxes: a tensor of bounding boxes.
      Can be batched or unbatched.

    Returns:
      bounding_boxes_tensor: a ragged tensor containing filtered bounding boxes.
    """
    if isinstance(bounding_boxes, tf.Tensor):
        bounding_boxes = tf.RaggedTensor.from_tensor(bounding_boxes)

    def drop_padded_boxes(bounding_boxes):
        bounding_boxes = bounding_boxes.to_tensor()
        mask = bounding_boxes[:, 4] != -1
        filtered_bounding_boxes = tf.boolean_mask(bounding_boxes, mask, axis=0)
        return tf.RaggedTensor.from_tensor(filtered_bounding_boxes)

    return tf.map_fn(drop_padded_boxes, bounding_boxes)


def add_class_id(bounding_boxes, class_id=0):
    """Add class ID to the innermost Tensor or RaggedTensor representing bounding boxes.

       Bounding box utilities in Keras_CV expects bounding boxes to have class IDs
       along with bounding box cordinates. This utility adds a class ID to the
       innermost tensor representing the bounding boxes.

       Usage example:
        bounding_boxes = tf.random.uniform(shape=[2, 2, 4])
        bounding_boxes_with_class_id = keras_cv.bounding_box.add_class_id(
                                        bounding_boxes, class_id=1)
        bounding_boxes_with_class_id is a Tensor of shape [2, 2, 5]

    Args:
      bounding_boxes: a tensor of bounding boxes.
        Can be batched or unbatched.
      class_id: The value of class id that needs to be padded.
        The default value is 0

    Returns:
      bounding_boxes_tensor: a tensor containing class id padded bounding boxes.
    """
    # format input bounding boxes
    is_ragged = isinstance(bounding_boxes, tf.RaggedTensor)
    if is_ragged:
        row_lengths = list(bounding_boxes.nested_row_lengths())
        row_lengths[1] = row_lengths[1] + 1
        dense_bounding_boxes = bounding_boxes.to_tensor()
    else:
        dense_bounding_boxes = bounding_boxes

    # pad input bounding boxes
    paddings = tf.constant([[0, 0], [0, 0], [0, 1]])
    padded_bounding_boxes = tf.pad(
        dense_bounding_boxes,
        paddings=paddings,
        mode="CONSTANT",
        constant_values=class_id,
    )

    # format output bounding boxes
    if isinstance(bounding_boxes, tf.RaggedTensor):
        padded_bounding_boxes = tf.RaggedTensor.from_tensor(
            padded_bounding_boxes,
            lengths=row_lengths,
        )
    return padded_bounding_boxes
