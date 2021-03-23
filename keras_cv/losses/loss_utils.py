"""Losses utilities for detection models."""

import tensorflow as tf


def multi_level_flatten(multi_level_inputs, last_dim=None):
  """Flattens a multi-level input.

  Args:
    multi_level_inputs: Ordered Dict with level to [batch, d1, ..., dm].
    last_dim: Whether the output should be [batch_size, None], or [batch_size,
      None, last_dim]. Defaults to `None`.

  Returns:
    Concatenated output [batch_size, None], or [batch_size, None, dm]
  """
  flattened_inputs = []
  batch_size = None
  for level in multi_level_inputs.keys():
    single_input = multi_level_inputs[level]
    if batch_size is None:
      batch_size = single_input.shape[0] or tf.shape(single_input)[0]
    if last_dim is not None:
      flattened_input = tf.reshape(single_input, [batch_size, -1, last_dim])
    else:
      flattened_input = tf.reshape(single_input, [batch_size, -1])
    flattened_inputs.append(flattened_input)
  return tf.concat(flattened_inputs, axis=1)
