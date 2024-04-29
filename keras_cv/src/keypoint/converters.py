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
"""Converter functions for working with keypoints formats."""

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export


# Internal exception
class _RequiresImagesException(Exception):
    pass


def _rel_xy_to_xy(keypoints, images=None):
    if images is None:
        raise _RequiresImagesException()
    shape = tf.cast(tf.shape(images), keypoints.dtype)
    h, w = shape[1], shape[2]

    x, y, rest = tf.split(keypoints, [1, 1, keypoints.shape[-1] - 2], axis=-1)

    return tf.concat([x * w, y * h, rest], axis=-1)


def _xy_to_rel_xy(keypoints, images=None):
    if images is None:
        raise _RequiresImagesException()
    shape = tf.cast(tf.shape(images), keypoints.dtype)
    h, w = shape[1], shape[2]

    x, y, rest = tf.split(keypoints, [1, 1, keypoints.shape[-1] - 2], axis=-1)

    return tf.concat([x / w, y / h, rest], axis=-1)


def _xy_noop(keypoints, images=None):
    return keypoints


TO_XY_CONVERTERS = {
    "xy": _xy_noop,
    "rel_xy": _rel_xy_to_xy,
}

FROM_XY_CONVERTERS = {
    "xy": _xy_noop,
    "rel_xy": _xy_to_rel_xy,
}


@keras_cv_export(
    "keras_cv.keypoint.convert_format", package="keras_cv.keypoint"
)
def convert_format(keypoints, source, target, images=None, dtype=None):
    """Converts keypoints from one format to another.

    Supported formats are:
    - `"xy"`, absolute pixel positions.
    - `"rel_xyxy"`. relative pixel positions.

    Formats are case-insensitive. It is recommended that you
    capitalize width and height to maximize the visual difference
    between `"xyWH"` and `"xyxy"`.

    Relative formats, abbreviated `rel`, make use of the shapes of the
    `images` passed. In these formats, the coordinates, widths, and
    heights are all specified as percentages of the host image.
    `images` may be a ragged Tensor. Note that using a ragged Tensor
    for images may cause a substantial performance loss, as each image
    will need to be processed separately due to the mismatching image
    shapes.

    Example:

    ```python
    images, keypoints = load_my_dataset()
    keypoints_in_rel = keras_cv.keypoint.convert_format(
        keypoint,
        source='xy',
        target='rel_xy',
        images=images,
    )
    ```

    Args:
        keypoints: tf.Tensor or tf.RaggedTensor representing keypoints
            in the format specified in the `source` parameter.
            `keypoints` can optionally have extra dimensions stacked
            on the final axis to store metadata. keypoints should
            have a rank between 2 and 4, with the shape
            `[num_boxes,*]`, `[batch_size, num_boxes, *]` or
            `[batch_size, num_groups, num_keypoints,*]`.
        source: One of {" ".join([f'"{f}"' for f in
            TO_XY_CONVERTERS.keys()])}. Used to specify the original
            format of the `boxes` parameter.
        target: One of {" ".join([f'"{f}"' for f in
            TO_XY_CONVERTERS.keys()])}. Used to specify the
            destination format of the `boxes` parameter.
        images: (Optional) a batch of images aligned with `boxes` on
            the first axis. Should be rank 3 (`HWC` format) or 4
            (`BHWC` format). Used in some converters to compute
            relative pixel values of the bounding box dimensions.
            Required when transforming from a rel format to a non-rel
            format.
        dtype: the data type to use when transforming the boxes.
            Defaults to None, i.e. `keypoints` dtype.
    """

    source = source.lower()
    target = target.lower()
    if source not in TO_XY_CONVERTERS:
        raise ValueError(
            f"convert_format() received an unsupported format for the argument "
            f"`source`. `source` should be one of {TO_XY_CONVERTERS.keys()}. "
            f"Got source={source}"
        )
    if target not in FROM_XY_CONVERTERS:
        raise ValueError(
            f"convert_format() received an unsupported format for the argument "
            f"`target`. `target` should be one of {FROM_XY_CONVERTERS.keys()}. "
            f"Got target={target}"
        )

    if dtype:
        keypoints = tf.cast(keypoints, dtype)

    if source == target:
        return keypoints

    keypoints, images, squeeze_axis = _format_inputs(keypoints, images)

    try:
        in_xy = TO_XY_CONVERTERS[source](keypoints, images=images)
        result = FROM_XY_CONVERTERS[target](in_xy, images=images)
    except _RequiresImagesException:
        raise ValueError(
            "convert_format() must receive `images` when transforming "
            f"between relative and absolute formats. "
            f"convert_format() received source=`{source}`, target=`{target}`, "
            f"but images={images}"
        )

    return _format_outputs(result, squeeze_axis)


def _format_inputs(keypoints, images):
    keypoints_rank = len(keypoints.shape)
    if keypoints_rank > 4:
        raise ValueError(
            "Expected keypoints rank to be in [2, 4], got "
            f"len(keypoints.shape)={keypoints_rank}."
        )
    keypoints_includes_batch = keypoints_rank > 2
    keypoints_are_grouped = keypoints_rank == 4
    if images is not None:
        images_rank = len(images.shape)
        if images_rank > 4 or images_rank < 3:
            raise ValueError(
                "Expected images rank to be 3 or 4, got "
                f"len(images.shape)={images_rank}."
            )
        images_include_batch = images_rank == 4
        if keypoints_includes_batch != images_include_batch:
            raise ValueError(
                "convert_format() expects both `keypoints` and `images` to be "
                "batched or both unbatched. Received "
                f"len(keypoints.shape)={keypoints_rank}, "
                f"len(images.shape)={images_rank}. Expected either "
                "len(keypoints.shape)=2 and len(images.shape)=3, or "
                "len(keypoints.shape)>=3 and len(images.shape)=4."
            )
        if not images_include_batch:
            images = tf.expand_dims(images, axis=0)

    squeeze_axis = []
    if not keypoints_includes_batch:
        keypoints = tf.expand_dims(keypoints, axis=0)
        squeeze_axis.append(0)
    if not keypoints_are_grouped:
        keypoints = tf.expand_dims(keypoints, axis=1)
        squeeze_axis.append(1)

    return keypoints, images, squeeze_axis


def _format_outputs(result, squeeze_axis):
    if len(squeeze_axis) == 0:
        return result
    return tf.squeeze(result, axis=squeeze_axis)
