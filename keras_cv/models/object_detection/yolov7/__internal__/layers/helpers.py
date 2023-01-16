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


class ReOrg(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ReOrg, self).__init__(**kwargs)

    def call(self, x):
        x = tf.concat(
            [
                x[:, ::2, ::2, :],
                x[:, 1::2, ::2, :],
                x[:, ::2, 1::2, :],
                x[..., 1::2, 1::2, :],
            ],
            -1,
        )
        return x


class Shortcut(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Shortcut, self).__init__(**kwargs)

    def call(self, x):
        return x[0] + x[1]
