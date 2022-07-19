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
"""Utility functions for keypoint transformation."""
import tensorflow as tf

H_AXIS = -3
W_AXIS = -2


def filter_out_of_image(keypoints, image):
    """Discards keypoints if falling outside of the image.

    Args:
      keypoints: a, possibly ragged, 2D (ungrouped), 3D (grouped)
        keypoint data in the 'xy' format.
      image: a 3D tensor in the HWC format.

    Returns:
      tf.RaggedTensor: a 2D or 3D ragged tensor with at least one
        ragged rank containing only keypoint in the image.
    """

    image_shape = tf.cast(tf.shape(image), keypoints.dtype)
    mask = tf.math.logical_and(
        tf.math.logical_and(
            keypoints[..., 0] >= 0, keypoints[..., 0] < image_shape[W_AXIS]
        ),
        tf.math.logical_and(
            keypoints[..., 1] >= 0, keypoints[..., 1] < image_shape[H_AXIS]
        ),
    )
    masked = tf.ragged.boolean_mask(keypoints, mask)
    if isinstance(masked, tf.RaggedTensor):
        return masked
    return tf.RaggedTensor.from_tensor(masked)
