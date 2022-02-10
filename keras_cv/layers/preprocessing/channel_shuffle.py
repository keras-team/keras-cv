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
from tensorflow.keras import layers, backend

class ChannelShuffle(layers.Layer):
    """ChannelShuffle class for shuffling the channel of RGB image. The expected images 
    should be [0-255] pixel ranges.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        seed:
            Integer. Used to create a random seed.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    channelshuffle = keras_cv.layers.preprocessing.ChannelShuffle()
    augmented_images = channelshuffle(images)
    ```
    """
    def __init__(
        self,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seed = seed

    @tf.function
    def _channel_shuffling(self, image):
        x = tf.transpose(image)
        x = tf.random.shuffle(x, seed=self.seed)
        return tf.transpose(x)

    def call(self, images, training=None):
        """call method for the ChannelShuffle layer.

        Args:
            images: Tensor representing images of shape
                [batch_size, width, height, channels], with dtype tf.float32 / tf.uint8, or,
                [width, height, channels], with dtype tf.float32 / tf.uint8
        Returns:
            images: augmented images, same shape as input.
        """
        if training is None:
            training = backend.learning_phase()

        if not training:
            return images
        else:
            unbatched = images.shape.rank == 3
            if unbatched:
                images = tf.expand_dims(images, axis=0)

            # TODO: Make the batch operation vectorize.
            output = tf.map_fn(lambda image: self._channel_shuffling(image), images)

            if unbatched:
                output = tf.squeeze(output, axis=0)
            return output

    def get_config(self):
        config = {
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape