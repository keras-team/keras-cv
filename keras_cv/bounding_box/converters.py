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

from typing import List
from typing import Optional
from typing import Union

import tensorflow as tf


# Internal exception to propagate the fact images was not passed to a converter that
# needs it
class RequiresImagesException(Exception):
    pass


def _encode_box_to_deltas(
    anchors: tf.Tensor,
    boxes: tf.Tensor,
    anchor_format: str,
    box_format: str,
    variance: Optional[Union[List[float], tf.Tensor]] = None,
):
    """Converts bounding_boxes from `center_yxhw` to delta format."""
    if variance is not None:
        if tf.is_tensor(variance):
            var_len = variance.get_shape().as_list()[-1]
        else:
            var_len = len(variance)
        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")
    encoded_anchors = convert_format(
        anchors,
        source=anchor_format,
        target="center_yxhw",
    )
    boxes = convert_format(
        boxes,
        source=box_format,
        target="center_yxhw",
    )
    anchor_dimensions = tf.maximum(encoded_anchors[..., 2:], tf.keras.backend.epsilon())
    box_dimensions = tf.maximum(boxes[..., 2:], tf.keras.backend.epsilon())
    # anchors be unbatched, boxes can either be batched or unbatched.
    boxes_delta = tf.concat(
        [
            (boxes[..., :2] - encoded_anchors[..., :2]) / anchor_dimensions,
            tf.math.log(box_dimensions / anchor_dimensions),
        ],
        axis=-1,
    )
    if variance is not None:
        boxes_delta /= variance
    return boxes_delta


def _decode_deltas_to_boxes(
    anchors: tf.Tensor,
    boxes_delta: tf.Tensor,
    anchor_format: str,
    box_format: str,
    variance: Optional[Union[List[float], tf.Tensor]] = None,
):
    """Converts bounding_boxes from delta format to `center_yxhw`."""
    if variance is not None:
        if tf.is_tensor(variance):
            var_len = variance.get_shape().as_list()[-1]
        else:
            var_len = len(variance)
        if var_len != 4:
            raise ValueError(f"`variance` must be length 4, got {variance}")
    tf.nest.assert_same_structure(anchors, boxes_delta)

    def decode_single_level(anchor, box_delta):
        encoded_anchor = convert_format(
            anchor,
            source=anchor_format,
            target="center_yxhw",
        )
        if variance is not None:
            box_delta = box_delta * variance
        # anchors be unbatched, boxes can either be batched or unbatched.
        box = tf.concat(
            [
                box_delta[..., :2] * encoded_anchor[..., 2:] + encoded_anchor[..., :2],
                tf.math.exp(box_delta[..., 2:]) * encoded_anchor[..., 2:],
            ],
            axis=-1,
        )
        box = convert_format(box, source="center_yxhw", target=box_format)
        return box

    if isinstance(anchors, dict) and isinstance(boxes_delta, dict):
        boxes = {}
        for lvl, anchor in anchors.items():
            boxes[lvl] = decode_single_level(anchor, boxes_delta[lvl])
        return boxes
    else:
        return decode_single_level(anchors, boxes_delta)


def _center_yxhw_to_xyxy(boxes, images=None, image_shape=None):
    y, x, height, width, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0, rest],
        axis=-1,
    )


def _center_xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [x - width / 2.0, y - height / 2.0, x + width / 2.0, y + height / 2.0, rest],
        axis=-1,
    )


def _xywh_to_xyxy(boxes, images=None, image_shape=None):
    x, y, width, height, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat([x, y, x + width, y + height, rest], axis=-1)


def _xyxy_to_center_yxhw(boxes, images=None, image_shape=None):
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [(top + bottom) / 2.0, (left + right) / 2.0, bottom - top, right - left, rest],
        axis=-1,
    )


def _rel_xywh_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    x, y, width, height, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [
            image_width * x,
            image_height * y,
            image_width * (x + width),
            image_height * (y + height),
            rest,
        ],
        axis=-1,
    )


def _xyxy_no_op(boxes, images=None, image_shape=None):
    return boxes


def _xyxy_to_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [left, top, right - left, bottom - top, rest],
        axis=-1,
    )


def _xyxy_to_rel_xywh(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = (
        left / image_width,
        right / image_width,
    )
    top, bottom = top / image_height, bottom / image_height
    return tf.concat(
        [left, top, right - left, bottom - top, rest],
        axis=-1,
    )


def _xyxy_to_center_xywh(boxes, images=None, image_shape=None):
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    return tf.concat(
        [(left + right) / 2.0, (top + bottom) / 2.0, right - left, bottom - top, rest],
        axis=-1,
    )


def _rel_xyxy_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _xyxy_to_rel_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _yxyx_to_xyxy(boxes, images=None, image_shape=None):
    y1, x1, y2, x2, rest = tf.split(boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1)
    return tf.concat([x1, y1, x2, y2, rest], axis=-1)


def _rel_yxyx_to_xyxy(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    top, left, bottom, right, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left * image_width, right * image_width
    top, bottom = top * image_height, bottom * image_height
    return tf.concat(
        [left, top, right, bottom, rest],
        axis=-1,
    )


def _xyxy_to_yxyx(boxes, images=None, image_shape=None):
    x1, y1, x2, y2, rest = tf.split(boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1)
    return tf.concat([y1, x1, y2, x2, rest], axis=-1)


def _xyxy_to_rel_yxyx(boxes, images=None, image_shape=None):
    image_height, image_width = _image_shape(images, image_shape, boxes)
    left, top, right, bottom, rest = tf.split(
        boxes, [1, 1, 1, 1, boxes.shape[-1] - 4], axis=-1
    )
    left, right = left / image_width, right / image_width
    top, bottom = top / image_height, bottom / image_height
    return tf.concat(
        [top, left, bottom, right, rest],
        axis=-1,
    )


TO_XYXY_CONVERTERS = {
    "xywh": _xywh_to_xyxy,
    "center_xywh": _center_xywh_to_xyxy,
    "center_yxhw": _center_yxhw_to_xyxy,
    "rel_xywh": _rel_xywh_to_xyxy,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _rel_xyxy_to_xyxy,
    "yxyx": _yxyx_to_xyxy,
    "rel_yxyx": _rel_yxyx_to_xyxy,
}

FROM_XYXY_CONVERTERS = {
    "xywh": _xyxy_to_xywh,
    "center_xywh": _xyxy_to_center_xywh,
    "center_yxhw": _xyxy_to_center_yxhw,
    "rel_xywh": _xyxy_to_rel_xywh,
    "xyxy": _xyxy_no_op,
    "rel_xyxy": _xyxy_to_rel_xyxy,
    "yxyx": _xyxy_to_yxyx,
    "rel_yxyx": _xyxy_to_rel_yxyx,
}


def convert_format(
    boxes, source, target, images=None, image_shape=None, dtype="float32"
):
    f"""Converts bounding_boxes from one format to another.

    Supported formats are:
    - `"xyxy"`, also known as `corners` format.  In this format the first four axes
        represent [left, top, right, bottom] in that order.
    - `"rel_xyxy"`.  In this format, the axes are the same as `"xyxy"` but the x
        coordinates are normalized using the image width, and the y axes the image
        height.  All values in `rel_xyxy` are in the range (0, 1).
    - `"xywh"`.  In this format the first four axes represent
        [left, top, width, height].
    - `"rel_xywh".  In this format the first four axes represent
        [left, top, width, height], just like `"xywh"`.  Unlike `"xywh"`, the values
        are in the range (0, 1) instead of absolute pixel values.
    - `"center_xyWH"`.  In this format the first two coordinates represent the x and y
        coordinates of the center of the bounding box, while the last two represent
        the width and height of the bounding box.
    - `"center_yxHW"`.  In this format the first two coordinates represent the y and x
        coordinates of the center of the bounding box, while the last two represent
        the height and width of the bounding box.
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
    if images is not None and image_shape is not None:
        raise ValueError(
            "convert_format() expects either `images` or `image_shape`, "
            f"but not both.  Received images={images} image_shape={image_shape}"
        )

    _validate_image_shape(image_shape)

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

    # rel->rel conversions should not require images
    if source.startswith("rel") and target.startswith("rel"):
        source = source.replace("rel_", "", 1)
        target = target.replace("rel_", "", 1)

    boxes, images, squeeze = _format_inputs(boxes, images)
    to_xyxy_fn = TO_XYXY_CONVERTERS[source]
    from_xyxy_fn = FROM_XYXY_CONVERTERS[target]

    try:
        in_xyxy = to_xyxy_fn(boxes, images=images, image_shape=image_shape)
        result = from_xyxy_fn(in_xyxy, images=images, image_shape=image_shape)
    except RequiresImagesException:
        raise ValueError(
            "convert_format() must receive `images` or `image_shape` when transforming "
            f"between relative and absolute formats."
            f"convert_format() received source=`{format}`, target=`{format}, "
            f"but images={images} and image_shape={image_shape}."
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


def _validate_image_shape(image_shape):
    # Escape early if image_shape is None and skip validation.
    if image_shape is None:
        return
    # tuple/list
    if isinstance(image_shape, (tuple, list)):
        if len(image_shape) != 3:
            raise ValueError(
                "image_shape should be of length 3, but got "
                f"image_shape={image_shape}"
            )
        return

    # tensor
    if isinstance(image_shape, tf.Tensor):
        if len(image_shape.shape) > 1:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        if image_shape.shape[0] != 3:
            raise ValueError(
                "image_shape.shape should be (3), but got "
                f"image_shape.shape={image_shape.shape}"
            )
        return

    # Warn about failure cases
    raise ValueError(
        "Expected image_shape to be either a tuple, list, Tensor.  "
        f"Received image_shape={image_shape}"
    )


def _format_outputs(boxes, squeeze):
    if squeeze:
        return tf.squeeze(boxes, axis=0)
    return boxes


def _image_shape(images, image_shape, boxes):
    if images is None and image_shape is None:
        raise RequiresImagesException()

    if image_shape is None:
        if not isinstance(images, tf.RaggedTensor):
            image_shape = tf.shape(images)
            height, width = image_shape[1], image_shape[2]
        else:
            height = tf.reshape(images.row_lengths(), (-1, 1))
            width = tf.reshape(tf.reduce_max(images.row_lengths(axis=2), 1), (-1, 1))
            if isinstance(boxes, tf.RaggedTensor):
                height = tf.expand_dims(height, axis=-1)
                width = tf.expand_dims(width, axis=-1)
    else:
        height, width = image_shape[0], image_shape[1]
    return tf.cast(height, boxes.dtype), tf.cast(width, boxes.dtype)
