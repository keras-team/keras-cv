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


def transform_to_standard_range(images, value_range):
    """transforms the input Tensor to range [0, 255].

    This function is intended to be used in preprocessing layers that
    rely upon color values.  This allows us to assume internally that
    the input tensor is always in the range [0, 255].

    Args:
        images: the set of images to transform to the [0, 255] range
        value_range: the value range of the images to transform

    Returns:
        a new Tensor with values in the range [0, 255]

    Usage:
    ```python
    images = keras_cv.utils.preprocessing.transform_to_standard_range(
        images,
        value_range
    )
    images = tf.math.minimum(images + 10, 255)
    images = keras_cv.utils.transform_to_value_range(images, value_range)
    ```
    """
    images = tf.cast(images, dtype=tf.float32)
    min_value, max_value = _unwrap_value_range(value_range)
    images = (images - min_value) / (max_value - min_value)
    return images * 255.0


def transform_to_value_range(images, value_range):
    """transforms input Tensor into value_range.

    This function is intended to be used in preprocessing layers that
    rely upon color values.  This allows us to assume internally that
    the input tensor is always in the range [0, 255].  This function
    should be used at the end of the function body before returning
    the input tensor to the user.

    Args:
        images: the set of images to transform to the value range
        value_range: the value range to transform into

    Returns:
        a new Tensor with values in the range [0, 255]

    Usage:
    ```python
    images = keras_cv.utils.preprocessing.transform_to_standard_range(
        images,
        value_range
    )
    images = tf.math.minimum(images + 10, 255)
    images = keras_cv.utils.transform_to_value_range(images, value_range)
    ```
    """
    min_value, max_value = _unwrap_value_range(value_range)
    images = images / 255.0
    images = images * max_value - min_value
    return images


def _unwrap_value_range(value_range):
    min_value, max_value = value_range
    min_value = tf.cast(min_value, dtype=tf.float32)
    max_value = tf.cast(max_value, dtype=tf.float32)
    return min_value, max_value
