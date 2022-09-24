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
      selected_indicators: [`num_samples`] or [batch_size, `num_samples`]
        integer Tensor, 1 for indicating the index is sampled, 0 for
        indicating the index is not sampled.
    """

    def balanced_single_sample(positive_match, negative_match):
        num_pos_matches = tf.reduce_sum(tf.cast(positive_match, tf.int32))
        num_neg_matches = tf.reduce_sum(tf.cast(negative_match, tf.int32))
        num_pos = tf.cast(num_samples * positive_fraction, tf.int32)
        # if there are not enough positive samples, obtain all
        num_pos = tf.minimum(num_pos, num_pos_matches)
        num_neg = num_samples - num_pos
        # if there are not enough negative samples, obtain all
        num_neg = tf.minimum(num_neg, num_neg_matches)

        # we choose to use random generator instead of random shuffle since the latter
        # does not work on GPU.

        # randomly generate positive values for positive match, and 0 for negative match
        random_pos = tf.random.uniform(tf.shape(positive_match), minval=0.0, maxval=1.0)
        zeros = tf.zeros_like(random_pos)
        random_pos = tf.where(positive_match, random_pos, zeros)
        # pick the top k values as random positive indices
        _, positive_indices = tf.math.top_k(random_pos, k=num_pos)

        # randomly generate positive values for negative match, and 0 for positive match
        random_neg = tf.random.uniform(tf.shape(negative_match), minval=0.0, maxval=1.0)
        random_neg = tf.where(negative_match, random_neg, zeros)
        # pick the top k values as random negative indices
        _, negative_indices = tf.math.top_k(random_neg, k=num_neg)

        selected_indices = tf.concat([positive_indices, negative_indices], axis=0)
        selected_indicators = tf.scatter_nd(
            tf.expand_dims(selected_indices, axis=-1),
            tf.ones_like(selected_indices, dtype=tf.int32),
            tf.shape(positive_match),
        )
        return selected_indicators

    if len(positive_matches.get_shape().as_list()) == 1:
        return balanced_single_sample(positive_matches, negative_matches)
    else:

        def balance_fn(args):
            pos, neg = args
            return balanced_single_sample(pos, neg)

        return tf.vectorized_map(balance_fn, (positive_matches, negative_matches))
