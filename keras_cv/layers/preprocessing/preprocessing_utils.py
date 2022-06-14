# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for preprocessing layers."""

import tensorflow as tf
from tensorflow.keras import backend


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        inputs = tf.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = tf.cast(inputs, dtype)
    return inputs


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            "Unknown `fill_mode` {}. Only `reflect`, `wrap`, "
            "`constant` and `nearest` are supported.".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )


def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).
    Args:
      angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank must
        be statically known (the shape is not `TensorShape(None)`).
      image_height: Height of the image(s) to be transformed.
      image_width: Width of the image(s) to be transformed.
      name: The name of the op.
    Returns:
      A tensor of shape (num_images, 8). Projective transforms which can be
        given to operation `image_projective_transform_v2`. If one row of
        transforms is [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the
        *output* point `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`.
    """
    with backend.name_scope(name or "rotation_matrix"):
        x_offset = (
            (image_width - 1)
            - (tf.cos(angles) * (image_width - 1) - tf.sin(angles) * (image_height - 1))
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (tf.sin(angles) * (image_width - 1) + tf.cos(angles) * (image_height - 1))
        ) / 2.0
        num_angles = tf.shape(angles)[0]
        return tf.concat(
            values=[
                tf.cos(angles)[:, None],
                -tf.sin(angles)[:, None],
                x_offset[:, None],
                tf.sin(angles)[:, None],
                tf.cos(angles)[:, None],
                y_offset[:, None],
                tf.zeros((num_angles, 2), tf.float32),
            ],
            axis=1,
        )


def transform(
    images,
    transforms,
    fill_mode="reflect",
    fill_value=0.0,
    interpolation="bilinear",
    output_shape=None,
    name=None,
):
    """Applies the given transform(s) to the image(s).
    Args:
      images: A tensor of shape
        `(num_images, num_rows, num_columns, num_channels)` (NHWC). The rank
        must be statically known (the shape is not `TensorShape(None)`).
      transforms: Projective transform matrix/matrices. A vector of length 8 or
        tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1,
        b2, c0, c1], then it maps the *output* point `(x, y)` to a transformed
        *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
        where `k = c0 x + c1 y + 1`. The transforms are *inverted* compared
        to the transform mapping input points to output points. Note that
        gradients are not backpropagated into transformation parameters.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      output_shape: Output dimension after the transform, `[height, width]`.
        If `None`, output is the same size as input image.
      name: The name of the op.
    Fill mode behavior for each valid value is as follows:
    - reflect (d c b a | a b c d | d c b a)
    The input is extended by reflecting about the edge of the last pixel.
    - constant (k k k k | a b c d | k k k k)
    The input is extended by filling all
    values beyond the edge with the same constant value k = 0.
    - wrap (a b c d | a b c d | a b c d)
    The input is extended by wrapping around to the opposite edge.
    - nearest (a a a a | a b c d | d d d d)
    The input is extended by the nearest pixel.
    Input shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.
    Output shape:
      4D tensor with shape: `(samples, height, width, channels)`,
        in `"channels_last"` format.
    Returns:
      Image(s) with the same type and shape as `images`, with the given
      transform(s) applied. Transformed coordinates outside of the input image
      will be filled with zeros.
    Raises:
      TypeError: If `image` is an invalid type.
      ValueError: If output shape is not 1-D int32 Tensor.
    """
    with backend.name_scope(name or "transform"):
        if output_shape is None:
            output_shape = tf.shape(images)[1:3]
            if not tf.executing_eagerly():
                output_shape_value = tf.get_static_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = tf.convert_to_tensor(output_shape, tf.int32, name="output_shape")

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                "{}".format(output_shape)
            )

        fill_value = tf.convert_to_tensor(fill_value, tf.float32, name="fill_value")

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )
