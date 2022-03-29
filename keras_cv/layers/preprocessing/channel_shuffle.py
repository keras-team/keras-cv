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
class ChannelShuffle(layers.Layer):
    """Shuffle channels of an input image.

    Input shape:
        The expected images should be [0-255] pixel ranges.
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        groups: Number of groups to divide the input channels. Default 3.
        seed: Integer. Used to create a random seed.

    Call arguments:
        images: Tensor representing images of shape
            `(batch_size, width, height, channels)`, with dtype tf.float32 / tf.uint8,
            ` or (width, height, channels)`, with dtype tf.float32 / tf.uint8
        training: A boolean argument that determines whether the call should be run
            in inference mode or training mode. Default: True.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    channel_shuffle = keras_cv.layers.ChannelShuffle()
    augmented_images = channel_shuffle(images)
    ```
    """

    def __init__(self, groups=3, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.seed = seed

    def _channel_shuffling(self, images):
        unbatched = images.shape.rank == 3

        if unbatched:
            images = tf.expand_dims(images, axis=0)

        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        num_channels = images.shape[3]

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )

        channels_per_group = num_channels // self.groups
        images = tf.reshape(
            images, [-1, height, width, self.groups, channels_per_group]
        )
        images = tf.transpose(images, perm=[3, 1, 2, 4, 0])
        images = tf.random.shuffle(images, seed=self.seed)
        images = tf.transpose(images, perm=[4, 1, 2, 3, 0])
        images = tf.reshape(images, [-1, height, width, num_channels])

        if unbatched:
            images = tf.squeeze(images, axis=0)

        return images

    def call(self, images, training=True):
        if training:
            return self._channel_shuffling(images)
        else:
            return images

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups, "seed": self.seed})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
