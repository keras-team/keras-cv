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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.bounding_box.formats import XYWH


@keras_cv_export("keras_cv.bounding_box.is_relative")
def is_relative(bounding_box_format):
    """A util to check if a bounding box format uses relative coordinates"""
    if (
        bounding_box_format.lower()
        not in bounding_box.converters.TO_XYXY_CONVERTERS
    ):
        raise ValueError(
            "`is_relative()` received an unsupported format for the argument "
            f"`bounding_box_format`. `bounding_box_format` should be one of "
            f"{bounding_box.converters.TO_XYXY_CONVERTERS.keys()}. "
            f"Got bounding_box_format={bounding_box_format}"
        )

    return bounding_box_format.startswith("rel")


@keras_cv_export("keras_cv.bounding_box.as_relative")
def as_relative(bounding_box_format):
    """A util to get the relative equivalent of a provided bounding box format.

    If the specified format is already a relative format,
    it will be returned unchanged.
    """

    if not is_relative(bounding_box_format):
        return "rel_" + bounding_box_format

    return bounding_box_format


def _relative_area(boxes, bounding_box_format):
    boxes = bounding_box.convert_format(
        boxes,
        source=bounding_box_format,
        target="rel_xywh",
    )
    widths = boxes[..., XYWH.WIDTH]
    heights = boxes[..., XYWH.HEIGHT]
    # handle corner case where shear performs a full inversion.
    return ops.where(
        ops.logical_and(widths > 0, heights > 0), widths * heights, 0.0
    )


@keras_cv_export("keras_cv.bounding_box.clip_to_image")
def clip_to_image(
    bounding_boxes, bounding_box_format, images=None, image_shape=None
):
    """clips bounding boxes to image boundaries.

    `clip_to_image()` clips bounding boxes that have coordinates out of bounds
    of an image down to the boundaries of the image. This is done by converting
    the bounding box to relative formats, then clipping them to the `[0, 1]`
    range. Additionally, bounding boxes that end up with a zero area have their
    class ID set to -1, indicating that there is no object present in them.

    Args:
        bounding_boxes: bounding box tensor to clip.
        bounding_box_format: the KerasCV bounding box format the bounding boxes
            are in.
        images: list of images to clip the bounding boxes to.
        image_shape: the shape of the images to clip the bounding boxes to.
    """
    boxes, classes = bounding_boxes["boxes"], bounding_boxes["classes"]

    boxes = bounding_box.convert_format(
        boxes,
        source=bounding_box_format,
        target="rel_xyxy",
        images=images,
        image_shape=image_shape,
    )
    boxes, classes, images, squeeze = _format_inputs(boxes, classes, images)
    x1, y1, x2, y2 = ops.split(boxes, 4, axis=-1)
    clipped_bounding_boxes = ops.concatenate(
        [
            ops.clip(x1, 0, 1),
            ops.clip(y1, 0, 1),
            ops.clip(x2, 0, 1),
            ops.clip(y2, 0, 1),
        ],
        axis=-1,
    )
    areas = _relative_area(
        clipped_bounding_boxes, bounding_box_format="rel_xyxy"
    )
    clipped_bounding_boxes = bounding_box.convert_format(
        clipped_bounding_boxes,
        source="rel_xyxy",
        target=bounding_box_format,
        images=images,
        image_shape=image_shape,
    )
    clipped_bounding_boxes = ops.where(
        ops.expand_dims(areas > 0.0, axis=-1), clipped_bounding_boxes, -1.0
    )
    classes = ops.where(areas > 0.0, classes, -1)
    nan_indices = ops.any(ops.isnan(clipped_bounding_boxes), axis=-1)
    classes = ops.where(nan_indices, -1, classes)

    # TODO update dict and return
    clipped_bounding_boxes, classes = _format_outputs(
        clipped_bounding_boxes, classes, squeeze
    )

    result = bounding_boxes.copy()
    result["boxes"] = clipped_bounding_boxes
    result["classes"] = classes
    return result


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
        image_shape = ops.cast(image_shape, dtype=boxes.dtype)
        height = image_shape[0]
        width = image_shape[1]
        max_length = ops.stack([height, width, height, width], axis=-1)

    clipped_boxes = ops.maximum(ops.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def _format_inputs(boxes, classes, images):
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
                "clip_to_image() expects both boxes and images to be batched, "
                "or both boxes and images to be unbatched. Received "
                f"len(boxes.shape)={boxes_rank}, "
                f"len(images.shape)={images_rank}. Expected either "
                "len(boxes.shape)=2 AND len(images.shape)=3, or "
                "len(boxes.shape)=3 AND len(images.shape)=4."
            )
        if not images_include_batch:
            images = ops.expand_dims(images, axis=0)

    if not boxes_includes_batch:
        return (
            ops.expand_dims(boxes, axis=0),
            ops.expand_dims(classes, axis=0),
            images,
            True,
        )
    return boxes, classes, images, False


def _format_outputs(boxes, classes, squeeze):
    if squeeze:
        return ops.squeeze(boxes, axis=0), ops.squeeze(classes, axis=0)
    return boxes, classes
