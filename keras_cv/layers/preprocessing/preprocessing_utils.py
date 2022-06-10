# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for preprocessing layers."""

import tensorflow as tf


def ensure_tensor(inputs, dtype=None):
    """Ensures the input is a Tensor, SparseTensor or RaggedTensor."""
    if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
        inputs = tf.convert_to_tensor(inputs, dtype)
    if dtype is not None and inputs.dtype != dtype:
        inputs = tf.cast(inputs, dtype)
    return inputs
