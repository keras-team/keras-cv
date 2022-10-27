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
from tensorflow import keras


def gather_channels(*matrices, indices=None):
    # Gather channel axis according to the indices.
    if indices is None:
        return matrices

    gathered_channels = []

    for matrix in matrices:
        if keras.backend.image_data_format() == "channels_last":
            matrix = tf.gather(matrix, indices, axis=-1)
        else:
            matrix = tf.gather(matrix, indices, axis=1)
        gathered_channels.append(matrix)

    return gathered_channels
