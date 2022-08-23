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

import numpy as np
import tensorflow as tf


def rle_to_mask2d(mask_rle, shape):
    """Converts a Run-length encoded segmentation map to a 2-dimensional segmentation map.

    Args:
      mask_rle: a string that represents a run-length encoded segmentation map.
      shape: shape of the resultant 2-dimensional segmentation map.

    Returns:
      mask: 2-dimensional segmentation map.
    """
    shape = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape)
    
    # Split string
    s = tf.strings.split(mask_rle)
    s = tf.strings.to_number(s, tf.int64)
    
    # Get starts and lengths
    starts = s[::2] - 1
    lengths = s[1::2]
    
    # Make ones to be scattered
    total_ones = tf.reduce_sum(lengths)
    ones = tf.ones([total_ones], tf.uint8)
    
    # Make scattering indices
    r = tf.range(total_ones)
    cumulative_sum_of_lengths = tf.math.cumsum(lengths)
    s = tf.searchsorted(cumulative_sum_of_lengths, r, "right")
    idx = r + tf.gather(
        starts - tf.pad(cumulative_sum_of_lengths[:-1], [(1, 0)]), s
    )  # Search where r goes in cumulative_sum_of_lengths
    
    # Scatter ones into flattened mask
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    
    # Transpose and Reshape into mask
    mask = tf.transpose(tf.reshape(mask_flat, shape))
    return mask


def mask2d_to_rle(mask):
    """Converts a 2-dimensional segmentation map to a Run-length encoded segmentation map.

    Args:
      mask: a tensor that represents a 2-dimensional segmentation map.

    Returns:
      mask_rle: a string that represents the run-length encoded segmentation map.
    """
    # Transpose the mask
    mask = tf.transpose(mask).numpy()
    
    # Get pixel values from the mask
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    
    # Get run-length encoding
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    
    # Join run-length encodings
    mask_rle = " ".join(str(x) for x in runs)
    return mask_rle
