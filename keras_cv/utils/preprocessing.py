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


def transform_value_range(images, original_range, target_range, dtype=tf.float32):
    """transforms values in input tensor from original_range to target_range.

    This function is intended to be used in preprocessing layers that
    rely upon color values.  This allows us to assume internally that
    the input tensor is always in the range [0, 255].

    Args:
        images: the set of images to transform to the target range range.
        original_range: the value range to transform from.
        target_range: the value range to transform to.
        dtype: the dtype to compute the conversion with.  Defaults to tf.float32.
    Returns:
        a new Tensor with values in the target range.

    Usage:
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
    images = tf.cast(images, dtype=dtype)
    original_min_value, original_max_value = _unwrap_value_range(
        original_range, dtype=dtype
    )
    target_min_value, target_max_value = _unwrap_value_range(target_range, dtype=dtype)

    # images in the [0, 1] scale
    images = (images - original_min_value) / (original_max_value - original_min_value)

    scale_factor = target_max_value - target_min_value
    return (images * scale_factor) + target_min_value


def _unwrap_value_range(value_range, dtype=tf.float32):
    min_value, max_value = value_range
    min_value = tf.cast(min_value, dtype=dtype)
    max_value = tf.cast(max_value, dtype=dtype)
    return min_value, max_value
