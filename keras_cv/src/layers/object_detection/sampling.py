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

from keras_cv.src.backend import ops
from keras_cv.src.backend import random


def balanced_sample(
    positive_matches,
    negative_matches,
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

    N = ops.shape(positive_matches)[-1]
    if N < num_samples:
        raise ValueError(
            "passed in {positive_matches.shape} has less element than "
            f"{num_samples}"
        )
    # random_val = tf.random.uniform(tf.shape(positive_matches), minval=0.,
    # maxval=1.)
    zeros = ops.zeros_like(positive_matches, dtype="float32")
    ones = ops.ones_like(positive_matches, dtype="float32")
    ones_rand = ones + random.uniform(ops.shape(ones), minval=-0.2, maxval=0.2)
    halfs = 0.5 * ops.ones_like(positive_matches, dtype="float32")
    halfs_rand = halfs + random.uniform(
        ops.shape(halfs), minval=-0.2, maxval=0.2
    )
    values = zeros
    values = ops.where(positive_matches, ones_rand, values)
    values = ops.where(negative_matches, halfs_rand, values)
    num_pos_samples = int(num_samples * positive_fraction)
    valid_matches = ops.logical_or(positive_matches, negative_matches)
    # this might contain negative samples as well
    _, positive_indices = ops.top_k(values, k=num_pos_samples)
    selected_indicators = ops.cast(
        ops.sum(ops.one_hot(positive_indices, N), axis=-2), dtype="bool"
    )
    # setting all selected samples to zeros
    values = ops.where(selected_indicators, zeros, values)
    # setting all excessive positive matches to zeros as well
    values = ops.where(positive_matches, zeros, values)
    num_neg_samples = num_samples - num_pos_samples
    _, negative_indices = ops.top_k(values, k=num_neg_samples)
    selected_indices = ops.concatenate(
        [positive_indices, negative_indices], axis=-1
    )
    selected_indicators = ops.sum(ops.one_hot(selected_indices, N), axis=-2)
    selected_indicators = ops.minimum(
        selected_indicators, ops.ones_like(selected_indicators)
    )
    selected_indicators = ops.where(
        valid_matches, selected_indicators, ops.zeros_like(selected_indicators)
    )
    return selected_indicators
