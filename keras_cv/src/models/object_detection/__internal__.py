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

import functools
import math

import numpy as np
import tensorflow as tf

try:
    import pandas as pd
except ImportError:
    pd = None


def unpack_input(data):
    if type(data) is dict:
        return data["images"], data["bounding_boxes"]
    else:
        return data


def _get_tensor_types():
    if pd is None:
        return (tf.Tensor, np.ndarray)
    else:
        return (tf.Tensor, np.ndarray, pd.Series, pd.DataFrame)


def convert_inputs_to_tf_dataset(
    x=None, y=None, sample_weight=None, batch_size=None
):
    if sample_weight is not None:
        raise ValueError("RetinaNet does not yet support `sample_weight`.")

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide a "
                f"value for `y` or `batch_size`. Got `y={y}`, "
                f"`batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size == "full":
        dataset = dataset.batch(x.shape[0])
    elif batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset


# TODO(lukewood): remove once exported from Keras core.
def train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

    The last part of data will become validation data.

    Args:
      arrays: Tensors to split. Allowed inputs are arbitrarily nested structures
        of Tensors and NumPy arrays.
      validation_split: Float between 0 and 1. The proportion of the dataset to
        include in the validation split. The rest of the dataset will be
        included in the training split.
    Returns:
      `(train_arrays, validation_arrays)`
    """

    def _can_split(t):
        tensor_types = _get_tensor_types()
        return isinstance(t, tensor_types) or t is None

    flat_arrays = tf.nest.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not _can_split(t)]
    if unsplitable:
        raise ValueError(
            "`validation_split` is only supported for Tensors or NumPy "
            "arrays, found following types in the input: {}".format(unsplitable)
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            "Training data contains {batch_dim} samples, which is not "
            "sufficient to split it into a validation and training set as "
            "specified by `validation_split={validation_split}`. Either "
            "provide more data, or a different value for the "
            "`validation_split` argument.".format(
                batch_dim=batch_dim, validation_split=validation_split
            )
        )

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]

    train_arrays = tf.nest.map_structure(
        functools.partial(_split, start=0, end=split_at), arrays
    )
    val_arrays = tf.nest.map_structure(
        functools.partial(_split, start=split_at, end=batch_dim), arrays
    )

    return train_arrays, val_arrays
