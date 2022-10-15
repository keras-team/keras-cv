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
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="keras_cv")
class ClassToken(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        learnable_class_token = tf.Variable(initial_value=tf.random.normal([1, 1, input_shape[-1]]))

        class_token_broadcast = tf.cast(
            tf.broadcast_to(learnable_class_token, [batch_size, 1, input_shape[-1]]),
            dtype=inputs.dtype,
        )
        return tf.concat([class_token_broadcast, inputs], 1)

    def get_config(self):
        base_config = super().get_config()
        return base_config