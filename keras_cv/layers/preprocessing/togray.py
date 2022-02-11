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
import tensorflow.keras.layers as layers
from tensorflow.keras import layers, backend


class ToGray(layers.Layer):
    """ToGray class for transforming RGB image to Grayscale image. The expected images 
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
    to_grayscale = keras_cv.layers.preprocessing.ToGray()
    augmented_images = to_grayscale(images)
    ```
    """

    def __init__(self, num_output_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.num_output_channels = num_output_channels

    def _check_input_params(self, num_output_channels):
        if num_output_channels not in [1, 3]:
            raise ValueError(
                f"Valid arugment for num_output_channels param are 1 or 3. Got {num_output_channels}"
            )
        self.num_output_channels = num_output_channels

    def _rgb_to_grayscale(self, image):
        if self.num_output_channels == 1:
            return tf.image.rgb_to_grayscale(image)
        elif self.num_output_channels == 3:
            _grayscale = tf.image.rgb_to_grayscale(image)
            return tf.concat([_grayscale, _grayscale, _grayscale], axis=-1)

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

        return tf.__internal__.smart_cond.smart_cond(
            training,
            true_fn=lambda: self._rgb_to_grayscale(images),
            false_fn=lambda: images,
        )

    def get_config(self):
        config = {
            "num_output_channels": self.num_output_channels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape