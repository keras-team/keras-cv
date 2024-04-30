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
from tensorflow import keras
from tensorflow.keras import backend

from keras_cv.src import core
from keras_cv.src.backend import ops

_TF_INTERPOLATION_METHODS = {
    "bilinear": tf.image.ResizeMethod.BILINEAR,
    "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    "bicubic": tf.image.ResizeMethod.BICUBIC,
    "area": tf.image.ResizeMethod.AREA,
    "lanczos3": tf.image.ResizeMethod.LANCZOS3,
    "lanczos5": tf.image.ResizeMethod.LANCZOS5,
    "gaussian": tf.image.ResizeMethod.GAUSSIAN,
    "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
}


def get_interpolation(interpolation):
    """fetches a valid interpolation method from `tf.image.ResizeMethod`.

    Args:
        interpolation: string representing an interpolation method.
    Raises:
        NotImplementedError: if the method passed is not recognized
    Returns:
        An interpolation method from `tf.image.ResizeMethod`

    """
    interpolation = interpolation.lower()
    if interpolation not in _TF_INTERPOLATION_METHODS:
        raise NotImplementedError(
            "Value not recognized for `interpolation`: {}. Supported values "
            "are: {}".format(interpolation, _TF_INTERPOLATION_METHODS.keys())
        )
    return _TF_INTERPOLATION_METHODS[interpolation]


def transform_value_range(
    images, original_range, target_range, dtype=tf.float32
):
    """transforms values in input tensor from original_range to target_range.
    This function is intended to be used in preprocessing layers that
    rely upon color values. This allows us to assume internally that
    the input tensor is always in the range [0, 255].

    Args:
        images: the set of images to transform to the target range.
        original_range: the value range to transform from.
        target_range: the value range to transform to.
        dtype: the dtype to compute the conversion with, defaults to tf.float32.

    Returns:
        a new Tensor with values in the target range.

    Example:
    ```python
    original_range = [0, 1]
    target_range = [0, 255]
    images = keras_cv.utils.preprocessing.transform_value_range(
        images,
        original_range,
        target_range
    )
    images = tf.math.minimum(images + 10, 255)
    images = keras_cv.utils.preprocessing.transform_value_range(
        images,
        target_range,
        original_range
    )
    ```
    """
    if (
        original_range[0] == target_range[0]
        and original_range[1] == target_range[1]
    ):
        return images

    images = tf.cast(images, dtype=dtype)
    original_min_value, original_max_value = _unwrap_value_range(
        original_range, dtype=dtype
    )
    target_min_value, target_max_value = _unwrap_value_range(
        target_range, dtype=dtype
    )

    # images in the [0, 1] scale
    images = (images - original_min_value) / (
        original_max_value - original_min_value
    )

    scale_factor = target_max_value - target_min_value
    return (images * scale_factor) + target_min_value


def _unwrap_value_range(value_range, dtype=tf.float32):
    min_value, max_value = value_range
    min_value = tf.cast(min_value, dtype=dtype)
    max_value = tf.cast(max_value, dtype=dtype)
    return min_value, max_value


def blend(image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
    """Blend image1 and image2 using 'factor'.

    FactorSampler should be in the range [0, 1]. A value of 0.0 means only
    image1 is used. A value of 1.0 means only image2 is used. A value between
    0.0 and 1.0 means we linearly interpolate the pixel values between the two
    images. A value greater than 1.0 "extrapolates" the difference
    between the two pixel values, and we clip the results to values
    between 0 and 255.
    Args:
      image1: An image Tensor of type tf.float32 with value range [0, 255].
      image2: An image Tensor of type tf.float32 with value range [0, 255].
      factor: A floating point value above 0.0.
    Returns:
      A blended image Tensor.
    """
    difference = image2 - image1
    scaled = factor * difference
    temp = image1 + scaled
    return tf.clip_by_value(temp, 0.0, 255.0)


def parse_factor(
    param, min_value=0.0, max_value=1.0, param_name="factor", seed=None
):
    if isinstance(param, dict):
        # For all classes missing a `from_config` implementation.
        # (RandomHue, RandomShear, etc.)
        # To be removed with addition of `keras.__internal__` namespace support
        param = keras.utils.deserialize_keras_object(param)

    if isinstance(param, core.FactorSampler):
        return param

    if isinstance(param, float) or isinstance(param, int):
        param = (min_value, param)

    if param[0] > param[1]:
        raise ValueError(
            f"`{param_name}[0] > {param_name}[1]`, `{param_name}[0]` must be "
            f"<= `{param_name}[1]`. Got `{param_name}={param}`"
        )
    if (min_value is not None and param[0] < min_value) or (
        max_value is not None and param[1] > max_value
    ):
        raise ValueError(
            f"`{param_name}` should be inside of range "
            f"[{min_value}, {max_value}]. Got {param_name}={param}"
        )

    if param[0] == param[1]:
        return core.ConstantFactorSampler(param[0])

    return core.UniformFactorSampler(param[0], param[1], seed=seed)


def random_inversion(random_generator):
    """Randomly returns a -1 or a 1 based on the provided random_generator.

    This can be used by KPLs to randomly invert sampled values.

    Args:
        random_generator: a Keras random number generator. An instance can be
            passed from the `self._random_generator` attribute of
            a `BaseImageAugmentationLayer`.

    Returns:
      either -1, or -1.
    """
    negate = random_generator.uniform((), 0, 1, dtype=tf.float32) > 0.5
    negate = tf.cond(negate, lambda: -1.0, lambda: 1.0)
    return negate


def batch_random_inversion(random_generator, batch_size):
    """Same as `random_inversion` but for batched inputs."""
    negate = random_generator.uniform((batch_size, 1), 0, 1, dtype=tf.float32)
    negate = tf.where(negate > 0.5, -1.0, 1.0)
    return negate


def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).
    Args:
      angles: A scalar angle to rotate all images by, or (for batches of images)
        a vector with an angle to rotate each image in the batch. The rank
        must be statically known (the shape is not `TensorShape(None)`).
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
            - (
                tf.cos(angles) * (image_width - 1)
                - tf.sin(angles) * (image_height - 1)
            )
        ) / 2.0
        y_offset = (
            (image_height - 1)
            - (
                tf.sin(angles) * (image_width - 1)
                + tf.cos(angles) * (image_height - 1)
            )
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


def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).
    Args:
      translations: A matrix of 2-element lists representing `[dx, dy]`
        to translate for each image (for a batch of images).
      name: The name of the op.
    Returns:
      A tensor of shape `(num_images, 8)` projective transforms which can be
        given to `transform`.
    """
    with backend.name_scope(name or "translation_matrix"):
        num_translations = tf.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return tf.concat(
            values=[
                tf.ones((num_translations, 1), tf.float32),
                tf.zeros((num_translations, 1), tf.float32),
                -translations[:, 0, None],
                tf.zeros((num_translations, 1), tf.float32),
                tf.ones((num_translations, 1), tf.float32),
                -translations[:, 1, None],
                tf.zeros((num_translations, 2), tf.float32),
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
        tensor of size N x 8. If one row of transforms is
        [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
        `(x, y)` to a transformed *input* point
        `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
        `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
        transform mapping input points to output points. Note that gradients are
        not backpropagated into transformation parameters.
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

        output_shape = tf.convert_to_tensor(
            output_shape, tf.int32, name="output_shape"
        )

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError(
                "output_shape must be a 1-D Tensor of 2 elements: "
                "new_height, new_width, instead got "
                "{}".format(output_shape)
            )

        fill_value = tf.convert_to_tensor(
            fill_value, tf.float32, name="fill_value"
        )

        return tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            output_shape=output_shape,
            fill_value=fill_value,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper(),
        )


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not ops.is_tensor(inputs):
        inputs = ops.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = ops.cast(inputs, dtype)
    return inputs


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {"reflect", "wrap", "constant", "nearest"}:
        raise NotImplementedError(
            " Want fillmode  to be one of `reflect`, `wrap`, "
            "`constant` or `nearest`. Got `fill_mode` {}. ".format(fill_mode)
        )
    if interpolation not in {"nearest", "bilinear"}:
        raise NotImplementedError(
            "Unknown `interpolation` {}. Only `nearest` and "
            "`bilinear` are supported.".format(interpolation)
        )
