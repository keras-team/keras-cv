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
"""Converter functions for working with bounding box formats."""

import tensorflow as tf


# Internal exception to propagate the fact images was not passed to a converter that
# needs it
class RequiresImagesException(Exception):
    pass


def _center_xywh_to_xyxy(boxes, images=None):
    x, y, width, height, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0, rest],
        axis=-1,
    )


def _xywh_to_xyxy(boxes, images=None):
    x, y, width, height, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat([x, y, x + width, y + height, rest], axis=-1)


def _xyxy_no_op(boxes, images=None):
    return boxes


def _xyxy_to_xywh(boxes, images=None):
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [left, top, right - left, bottom - top, rest],
        axis=-1,
    )


def _xyxy_to_center_xywh(boxes, images=None):
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [(left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top, rest],
        axis=-1,
    )


def _rel_xyxy_to_xyxy(boxes, images=None):
    if images is None:
        raise RequiresImagesException()
    shape = tf.shape(images)
    height, width = shape[1], shape[2]
    height, width = tf.cast(height, boxes.dtype), tf.cast(width, boxes.dtype)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left * width, right * width
    top, bottom = top * height, bottom * height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _xyxy_to_rel_xyxy(boxes, images=None):
    if images is None:
        raise RequiresImagesException()
    shape = tf.shape(images)
    height, width = shape[1], shape[2]
    height, width = tf.cast(height, boxes.dtype), tf.cast(width, boxes.dtype)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left / width, right / width
    top, bottom = top / height, bottom / height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _yxyx_to_xyxy(boxes, images=None):
    y1, x1, y2, x2, rest = tf.split(boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1)
    return tf.concat([x1, y1, x2, y2, rest], axis=-1)


def _rel_yxyx_to_xyxy(boxes, images=None):
    if images is None:
        raise RequiresImagesException()
    shape = tf.shape(images)
    height, width = shape[1], shape[2]
    height, width = tf.cast(height, boxes.dtype), tf.cast(width, boxes.dtype)
    top, left, bottom, right, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left * width, right * width
    top, bottom = top * height, bottom * height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _xyxy_to_yxyx(boxes, images=None):
    x1, y1, x2, y2, rest = tf.split(boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1)
    return tf.concat([y1, x1, y2, x2, rest], axis=-1)


def _xyxy_to_rel_yxyx(boxes, images=None):
    if images is None:
        raise RequiresImagesException()
    shape = tf.shape(images)
    height, width = shape[1], shape[2]
    height, width = tf.cast(height, boxes.dtype), tf.cast(width, boxes.dtype)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left / width, right / width
    top, bottom = top / height, bottom / height
    return tf.concat(
        [top, left, bottom, right, rest],
        axis=-1,
    )


TO_XYXY_CONVERTERS = {
    "xywh": _xywh_to_xyxy,
    "center_xywh": _center_xywh_to_xyxy,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _rel_xyxy_to_xyxy,
    "yxyx": _yxyx_to_xyxy,
    "rel_yxyx": _rel_yxyx_to_xyxy,
}

FROM_XYXY_CONVERTERS = {
    "xywh": _xyxy_to_xywh,
    "center_xywh": _xyxy_to_center_xywh,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _xyxy_to_rel_xyxy,
    "yxyx": _xyxy_to_yxyx,
    "rel_yxyx": _xyxy_to_rel_yxyx,
}


def convert_format(boxes, source, target, images=None, dtype="float32"):
    f"""Converts bounding_boxes from one format to another.

    Supported formats are:
    - `"xyxy"`, also known as `corners` format.  In this format the first four axes
        represent [left, top, right, bottom] in that order.
    - `"rel_xyxy"`.  In this format, the axes are the same as `"xyxy"` but the x
        coordinates are normalized using the image width, and the y axes the image
        height.  All values in `rel_xyxy` are in the range (0, 1).
    - `"xyWH"`.  In this format the first four axes represent
        [left, top, width, height].
    - `"center_xyWH"`.  In this format the first two coordinates represent the x and y
        coordinates of the center of the bounding box, while the last two represent
        the width and height of the bounding box.
    - `"yxyx"`.  In this format the first four axes represent [top, left, bottom, right]
        in that order.
    - `"rel_yxyx"`.  In this format, the axes are the same as `"yxyx"` but the x
        coordinates are normalized using the image width, and the y axes the image
        height.  All values in `rel_yxyx` are in the range (0, 1).
    Formats are case insensitive.  It is recommended that you capitalize width and
    height to maximize the visual difference between `"xyWH"` and `"xyxy"`.

    Relative formats, abbreviated `rel`, make use of the shapes of the `images` passed.
    In these formats, the coordinates, widths, and heights are all specified as
    percentages of the host image.  `images` may be a ragged Tensor.  Note that using a
    ragged Tensor for images may cause a substantial performance loss, as each image
    will need to be processed separately due to the mismatching image shapes.

    Usage:

    ```python
    boxes = load_coco_dataset()
    boxes_in_xywh = keras_cv.bounding_box.convert_format(
        boxes,
        source='xyxy',
        target='xyWH'
    )
    ```

    Args:
        boxes: tf.Tensor representing bounding boxes in the format specified in the
            `source` parameter.  `boxes` can optionally have extra dimensions stacked on
             the final axis to store metadata.  boxes should be a 3D Tensor, with the
             shape `[batch_size, num_boxes, *]`.
        source: One of {" ".join([f'"{f}"' for f in TO_XYXY_CONVERTERS.keys()])}.  Used
            to specify the original format of the `boxes` parameter.
        target: One of {" ".join([f'"{f}"' for f in TO_XYXY_CONVERTERS.keys()])}.  Used
            to specify the destination format of the `boxes` parameter.
        images: (Optional) a batch of images aligned with `boxes` on the first axis.
            Should be at least 3 dimensions, with the first 3 dimensions representing:
            `[batch_size, height, width]`.  Used in some converters to compute relative
            pixel values of the bounding box dimensions.  Required when transforming
            from a rel format to a non-rel format.
        dtype: the data type to use when transforming the boxes.  Defaults to
            `tf.float32`.
    """

    source = source.lower()
    target = target.lower()
    if source not in TO_XYXY_CONVERTERS:
        raise ValueError(
            f"`convert_format()` received an unsupported format for the argument "
            f"`source`.  `source` should be one of {TO_XYXY_CONVERTERS.keys()}. "
            f"Got source={source}"
        )
    if target not in FROM_XYXY_CONVERTERS:
        raise ValueError(
            f"`convert_format()` received an unsupported format for the argument "
            f"`target`.  `target` should be one of {FROM_XYXY_CONVERTERS.keys()}. "
            f"Got target={target}"
        )

    boxes = tf.cast(boxes, dtype)
    if source == target:
        return boxes

    boxes, images, squeeze = _format_inputs(boxes, images)

    to_xyxy_fn = TO_XYXY_CONVERTERS[source]
    from_xyxy_fn = FROM_XYXY_CONVERTERS[target]

    try:
        in_xyxy = to_xyxy_fn(boxes, images=images)
        result = from_xyxy_fn(in_xyxy, images=images)
    except RequiresImagesException:
        raise ValueError(
            "convert_format() must receive `images` when transforming "
            f"between relative and absolute formats."
            f"convert_format() received source=`{format}`, target=`{format}, "
            f"but images={images}"
        )

    return _format_outputs(result, squeeze)


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
                "convert_format() expects both boxes and images to be batched, or both "
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
