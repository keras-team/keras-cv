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


def HasRank(tensor, expected_rank):
    """Syntactic sugar for asserting that tensor has the expected rank."""
    if tensor.shape.ndims is not None and isinstance(expected_rank, int):
        assert tensor.shape.ndims == expected_rank, (
            "Ranks did not match, got %d, " "expected %d"
        ) % (tensor.shape.ndims, expected_rank)
    return tensor


def PadOrTrimTo(x, shape, pad_val=0, pad_after_contents=True):
    """Pad and slice x to the given shape.

    This is branched from Lingvo https://github.com/tensorflow/lingvo/blob/master/lingvo/core/py_utils.py.

    Args:
      x: A tensor.
      shape: The shape of the returned tensor.
      pad_val: An int or float used to pad x.
      pad_after_contents: Whether to pad and trim after the original contents of
        each dimension.
    Returns:
      'x' is padded with pad_val and sliced so that the result has the given
      shape.
    Raises:
      ValueError: if shape is a tf.TensorShape and not fully defined.
    """
    if isinstance(shape, (list, tuple)):
        expected_rank = len(shape)
    elif isinstance(shape, tf.TensorShape):
        if not shape.is_fully_defined():
            raise ValueError("shape %s padding %s must be fully defined." % (shape, x))
        expected_rank = shape.rank
    else:
        shape = HasRank(shape, 1)
        expected_rank = tf.size(shape)
    x = HasRank(x, expected_rank)

    pad = shape - tf.minimum(tf.shape(x), shape)
    zeros = tf.zeros_like(pad)
    if pad_after_contents:
        # If dim_i is less than shape[i], pads after contents.
        paddings = tf.stack([zeros, pad], axis=1)
        # If dim_i is larger than shape[i], we slice [0:shape[i]] for dim_i.
        slice_begin = zeros
    else:
        # If dim_i is less than shape[i], pads before contents.
        paddings = tf.stack([pad, zeros], axis=1)
        # If dim-i is larger than shape[i], we slice [dim_i - shape[i]:dim_i]
        # for dim_i.
        slice_begin = tf.shape(x) + pad - shape

    x = tf.pad(x, paddings, constant_values=pad_val)
    x = tf.slice(x, slice_begin, shape)

    return tf.reshape(x, shape)
