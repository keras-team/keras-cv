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
from keras_cv.bounding_box.formats import XYWH


def preserve_rel(target_bounding_box_format, bounding_box_format):
    """A util to add "rel_" to target_bounding_box_format for relative bounding_box_format"""
    if bounding_box_format.lower() not in bounding_box.converters.TO_XYXY_CONVERTERS:
        raise ValueError(
            "`preserve_rel()` received an unsupported format for the argument "
            f"`bounding_box_format`.  `bounding_box_format` should be one of "
            f"{bounding_box.converters.TO_XYXY_CONVERTERS.keys()}. "
            f"Got bounding_box_format={bounding_box_format}"
        )

    if target_bounding_box_format.startswith("rel"):
        raise ValueError(
            'Expected "target_bounding_box_format" to be non-relative. '
            f"Got `target_bounding_box_format`={target_bounding_box_format}."
        )
    if bounding_box_format.startswith("rel"):
        return "rel_" + target_bounding_box_format
    return target_bounding_box_format


def _relative_area(bounding_boxes, bounding_box_format, images):
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="rel_xywh",
        images=images,
    )
    widths = bounding_boxes[..., XYWH.WIDTH]
    heights = bounding_boxes[..., XYWH.HEIGHT]
    # handle corner case where shear performs a full inversion.
    return tf.where(tf.math.logical_and(widths > 0, heights > 0), widths * heights, 0.0)


def clip_to_image(bounding_boxes, bounding_box_format, images=None, image_shape=None):
    """clips bounding boxes to image boundaries.

    `clip_to_image()` clips bounding boxes that have coordinates out of bounds of an
    image down to the boundaries of the image.  This is done by converting the bounding
    box to relative formats, then clipping them to the `[0, 1]` range.  Additionally,
    bounding boxes that end up with a zero area have their class ID set to -1,
    indicating that there is no object present in them.

    Args:
        bounding_boxes: bounding box tensor to clip.
        bounding_box_format: the KerasCV bounding box format the bounding boxes are in.
        images: list of images to clip the bounding boxes to.
        image_shape: the shape of the images to clip the bounding boxes to.
    """
    if bounding_boxes.shape[-1] < 5:
        raise ValueError(
            "`bounding_boxes` must include a class_id index on the final "
            "axis.  This is used to set `bounding_boxes` that are fully outside of the "
            "provided image to the background class, -1."
        )
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="rel_xyxy",
        images=images,
        image_shape=image_shape,
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
    areas = _relative_area(
        clipped_bounding_boxes, bounding_box_format="rel_xyxy", images=images
    )
    clipped_bounding_boxes = bounding_box.convert_format(
        clipped_bounding_boxes,
        source="rel_xyxy",
        target=bounding_box_format,
        images=images,
        image_shape=image_shape,
    )
    clipped_bounding_boxes = tf.where(
        tf.expand_dims(areas > 0.0, axis=-1), clipped_bounding_boxes, -1.0
    )
    nan_indices = tf.math.reduce_any(tf.math.is_nan(clipped_bounding_boxes), axis=-1)
    clipped_bounding_boxes = tf.where(
        tf.expand_dims(nan_indices, axis=-1), -1.0, clipped_bounding_boxes
    )
    clipped_bounding_boxes = _format_outputs(clipped_bounding_boxes, squeeze)
    return clipped_bounding_boxes


# TODO (tanzhenyu): merge with clip_to_image
def _clip_boxes(boxes, box_format, image_shape):
    """Clip boxes to the boundaries of the image shape"""
    if boxes.shape[-1] != 4:
        raise ValueError(
            "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1])
        )

    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
        height, width, _ = image_shape
        max_length = [height, width, height, width]
    else:
        image_shape = tf.cast(image_shape, dtype=boxes.dtype)
        height, width, _ = tf.unstack(image_shape, axis=-1)
        max_length = tf.stack([height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


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


def pad_with_sentinels(bounding_boxes, sentinel_value=-1):
    """Pads the given bounding box tensor with sentinel_value.

    This is done to convert RaggedTensors into standard Dense
    tensors, which have better performance and compatibility
    within the TensorFlow ecosystem.

    Args:
        bounding_boxes: a ragged tensor of bounding boxes.
            Can be batched or unbatched.
        sentinel_value: Value to set for indices not specified
            in bounding_boxes. Defaults to -1.

    Returns:
        a Tensor containing the sentinel_value padded bounding boxes.
    """
    return bounding_boxes.to_tensor(default_value=sentinel_value)


def filter_sentinels(bounding_boxes, sentinel_value=-1):
    """converts a Dense padded bounding box `tf.Tensor` to a `tf.RaggedTensor`.

    Bounding boxes are ragged tensors in most use cases. Converting them to a dense
    tensor makes it easier to work with Tensorflow ecosystem.
    This function can be used to filter out the padded bounding boxes by
    checking for padded sentinel value of the class_id axis of the bounding_boxes.

    Args:
        bounding_boxes: a Tensor of bounding boxes.  May be batched, or unbatched.
        sentinel_value: Value used to filter dense bounding box tensor.
            bounding_boxes with class_id equal to sentinel_value will be dropped.

    Returns:
        `tf.RaggedTensor`or 'tf.Tensor' containing the filtered bounding boxes.
    """
    is_ragged = isinstance(bounding_boxes, tf.RaggedTensor)
    if is_ragged:
        bounding_boxes = bounding_box.pad_with_sentinels(
            bounding_boxes, sentinel_value=sentinel_value
        )
    mask = bounding_boxes[..., 4] != sentinel_value
    filtered_bounding_boxes = tf.ragged.boolean_mask(bounding_boxes, mask)
    return filtered_bounding_boxes


def add_class_id(bounding_boxes, class_id=0):
    """Add class ID to a new dimension of the final axis of a bounding box Tensor.

    Bounding box utilities in KerasCV expect bounding boxes to have class IDs.
    This utility adds a class ID to a new axis of the provided tf.Tensor.

    Usage:
    ```python
    bounding_boxes = tf.random.uniform(shape=[2, 2, 4])
    bounding_boxes_with_class_id = keras_cv.bounding_box.add_class_id(
                                    bounding_boxes, class_id=1)
    # bounding_boxes_with_class_id is a Tensor of shape [2, 2, 5]
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
            "The number of values along the final axis of `bounding_boxes` is "
            "expected to be 4. But got {}.".format(bounding_boxes.shape[-1])
        )
    bounding_box_rank = len(tf.shape(bounding_boxes))
    if bounding_box_rank == 2:
        paddings = tf.constant([[0, 0], [0, 1]])
    elif bounding_box_rank == 3:
        paddings = tf.constant([[0, 0], [0, 0], [0, 1]])
    else:
        raise ValueError(
            f"`bounding_boxes` should be of rank 2 or 3. However "
            f"add_class_id received `bounding_boxes` of rank={bounding_box_rank}"
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
