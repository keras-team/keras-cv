# Copyright 2023 The KerasCV Authors
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

from keras_cv.backend import config
from keras_cv.backend import keras
from keras_cv.backend import ops


def _decode_strings_to_utf8(inputs):
    """Recursively decodes to list of strings with 'utf-8' encoding."""
    if isinstance(inputs, bytes):
        # Handles the case when the input is a scalar string.
        return inputs.decode("utf-8", errors="ignore")
    else:
        # Recursively iterate when input is a list.
        return [_decode_strings_to_utf8(x) for x in inputs]


def tensor_to_list(inputs):
    """Converts a tensor to nested lists.

    Args:
        inputs: Input tensor, or dict/list/tuple of input tensors.
    """
    if not isinstance(inputs, (tf.RaggedTensor, tf.Tensor)):
        inputs = tf.convert_to_tensor(inputs)
    if isinstance(inputs, tf.RaggedTensor):
        list_outputs = inputs.to_list()
    elif isinstance(inputs, tf.Tensor):
        list_outputs = inputs.numpy()
        if inputs.shape.rank != 0:
            list_outputs = list_outputs.tolist()
    if inputs.dtype == tf.string:
        list_outputs = _decode_strings_to_utf8(list_outputs)
    return list_outputs


def convert_to_backend_tensor_or_python_list(x):
    """
    Convert a tensor to the backend friendly representation of the data.

    This wraps `ops.convert_to_tensor` to account for the fact that torch and
    jax both lack native types for ragged and string data.

    If we encounter one of these types in torch or jax, we will instead covert
    the tensor to simple pythonic types (lists of strings).
    """
    if isinstance(x, tf.RaggedTensor) or getattr(x, "dtype", None) == tf.string:
        return tensor_to_list(x)
    return ops.convert_to_tensor(x)


def convert_to_ragged_batch(inputs):
    """Convert pythonic or numpy-like input to a 2-D `tf.RaggedTensor`.

    This is useful for text preprocessing layers which deal with already
    tokenized or split text.

    Args:
        inputs: A pythonic or numpy-like input to covert. This input should
            represent a possibly batched list of token sequences.

    Returns:
        An `(inputs, unbatched, rectangular)` tuple, where `inputs` is a
        2-D `tf.RaggedTensor`, `unbatched` is `True` if the inputs were
        origianlly rank 1, and `rectangular` is `True` if the inputs rows are
        all of equal lengths.
    """
    # `tf.keras.layers.Layer` does a weird conversion in __call__, where a list
    # of lists of ints will become a list of list of scalar tensors. We could
    # clean this up if we no longer need to care about that case.
    if isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], (list, tuple)):
            rectangular = len(set([len(row) for row in inputs])) == 1
            rows = [
                tf.convert_to_tensor(row, dtype_hint="int32") for row in inputs
            ]
            inputs = tf.ragged.stack(rows).with_row_splits_dtype("int64")
        else:
            inputs = tf.convert_to_tensor(inputs)
            rectangular = True
    elif isinstance(inputs, tf.Tensor):
        rectangular = True
    elif isinstance(inputs, tf.RaggedTensor):
        rectangular = False
    elif hasattr(inputs, "__array__"):
        inputs = tf.convert_to_tensor(ops.convert_to_numpy(inputs))
        rectangular = True
    else:
        raise ValueError(
            f"Unknown tensor type. Tensor input can be passed as "
            "tensors, numpy arrays, or python lists. Received: "
            f"`type(inputs)={type(inputs)}`"
        )
    if inputs.shape.rank < 1 or inputs.shape.rank > 2:
        raise ValueError(
            f"Tokenized tensor input should be rank 1 (unbatched) or "
            f"rank 2 (batched). Received: `inputs.shape={input.shape}`"
        )
    unbatched = inputs.shape.rank == 1
    rectangular = rectangular or unbatched
    if unbatched:
        inputs = tf.expand_dims(inputs, 0)
    if isinstance(inputs, tf.Tensor):
        inputs = tf.RaggedTensor.from_tensor(inputs)
    return inputs, unbatched, rectangular


def truncate_at_token(inputs, token, mask):
    """Truncate at first instance of `token`, ignoring `mask`."""
    matches = (inputs == token) & (~mask)
    end_indices = tf.cast(tf.math.argmax(matches, -1), "int32")
    end_indices = tf.where(end_indices == 0, tf.shape(inputs)[-1], end_indices)
    return tf.RaggedTensor.from_tensor(inputs, end_indices)


def assert_tf_backend(symbol_name):
    if config.backend() != "tensorflow":
        raise RuntimeError(
            f"{symbol_name} requires the `tensorflow` backend. "
            "Please set `KERAS_BACKEND=tensorflow` when running your program."
        )


def is_tensor_type(x):
    return hasattr(x, "__array__")


def standardize_dtype(dtype):
    if config.keras_3():
        return keras.backend.standardize_dtype(dtype)
    if hasattr(dtype, "name"):
        return dtype.name
    return dtype


def is_float_dtype(dtype):
    return "float" in standardize_dtype(dtype)


def is_int_dtype(dtype):
    return "int" in standardize_dtype(dtype)


def is_string_dtype(dtype):
    return "string" in standardize_dtype(dtype)
