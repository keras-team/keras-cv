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

from keras_cv.src.backend import ops


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


def convert_inputs_to_list_of_tensor_segments(x):
    """Converts user inputs to a list of a tensor segments.

    For models and layers which accept lists of string tensors to pack together,
    this method converts user inputs to a uniform format in a way that can be
    considered canonical for the library.

    We handle the following:

    - A single string will be converted to a tensor and wrapped in a list.
    - A list of strings will be converted to a tensor and wrapped in a list.
    - A single tensor will be wrapped in a list.
    - A list of tensors will be passed through unaltered.

    All other inputs will result in an error. This effectively means that users
    who would like to pack multiple segments together should convert those
    segments to tensors before calling the layer. This removes any ambiguity
    in the input for those cases.
    """
    # Check the input type.
    is_string = isinstance(x, (str, bytes))
    is_tensor = hasattr(x, "__array__")
    is_string_list = (
        isinstance(x, (list, tuple)) and x and isinstance(x[0], (str, bytes))
    )
    is_tensor_list = (
        isinstance(x, (list, tuple)) and x and hasattr(x[0], "__array__")
    )

    if is_string or is_string_list:
        # Automatically convert raw strings or string lists to tensors.
        # Wrap this input as a single (possibly batched) segment.
        x = [tf.convert_to_tensor(x)]
    elif is_tensor:
        # Automatically wrap a single tensor as a single segment.
        x = [x]
    elif is_tensor_list:
        # Pass lists of tensors though unaltered.
        x = x
    else:
        # Error for all other input.
        raise ValueError(
            f"Unsupported input for `x`. `x` should be a string, a list of "
            "strings, or a list of tensors. If passing multiple segments "
            "which should packed together, please convert your inputs to a "
            f"list of tensors. Received `x={x}`"
        )
    return x
