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
import numpy as np
import tensorflow as tf

from keras_cv.src.backend import ops


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, tf.RaggedTensor):
        x = x.to_tensor(-1)
    x = ops.convert_to_numpy(x)
    # Important for consistency when working with visualization utilities
    return np.ascontiguousarray(x)
