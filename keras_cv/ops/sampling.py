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


def balanced_sample(
    positive_matches: tf.Tensor,
    negative_matches: tf.Tensor,
    num_samples: int,
    positive_fraction: float,
):
    """
    Sampling ops to balance positive and negative samples, deals with both
    batched and unbatched inputs.

    Args:
      positive_matches: [N] or [batch_size, N] boolean Tensor, True for
        indicating the index is a positive sample
      negative_matches: [N] or [batch_size, N] boolean Tensor, True for
        indicating the index is a negative sample
      num_samples: int, representing the number of samples to collect
      positive_fraction: float. 0.5 means positive samples should be half
        of all collected samples.

    Returns:
      selected_indicators: [N] or [batch_size, N]
        integer Tensor, 1 for indicating the index is sampled, 0 for
        indicating the index is not sampled.
    """

    N = positive_matches.get_shape().as_list()[-1]
    if N < num_samples:
        raise ValueError(
            f"passed in {positive_matches.shape} has less element than {num_samples}"
        )
    # random_val = tf.random.uniform(tf.shape(positive_matches), minval=0., maxval=1.)
    zeros = tf.zeros_like(positive_matches, dtype=tf.float32)
    ones = tf.ones_like(positive_matches, dtype=tf.float32)
    ones_rand = ones + tf.random.uniform(ones.shape, minval=-0.2, maxval=0.2)
    halfs = 0.5 * tf.ones_like(positive_matches, dtype=tf.float32)
    halfs_rand = halfs + tf.random.uniform(halfs.shape, minval=-0.2, maxval=0.2)
    values = zeros
    values = tf.where(positive_matches, ones_rand, values)
    values = tf.where(negative_matches, halfs_rand, values)
    num_pos_samples = int(num_samples * positive_fraction)
    valid_matches = tf.logical_or(positive_matches, negative_matches)
    # this might contain negative samples as well
    _, positive_indices = tf.math.top_k(values, k=num_pos_samples)
    selected_indicators = tf.cast(
        tf.reduce_sum(tf.one_hot(positive_indices, depth=N), axis=-2), tf.bool
    )
    # setting all selected samples to zeros
    values = tf.where(selected_indicators, zeros, values)
    # setting all excessive positive matches to zeros as well
    values = tf.where(positive_matches, zeros, values)
    num_neg_samples = num_samples - num_pos_samples
    _, negative_indices = tf.math.top_k(values, k=num_neg_samples)
    selected_indices = tf.concat([positive_indices, negative_indices], axis=-1)
    selected_indicators = tf.reduce_sum(tf.one_hot(selected_indices, depth=N), axis=-2)
    selected_indicators = tf.minimum(
        selected_indicators, tf.ones_like(selected_indicators)
    )
    selected_indicators = tf.where(
        valid_matches, selected_indicators, tf.zeros_like(selected_indicators)
    )
    return selected_indicators
