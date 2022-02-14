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


class Grayscale(layers.Layer):
    """Grayscale is a preprocessing layer that transforms RGB images to Grayscale images. 
    Input images should have values in the range of [0, 255].
    
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Args:
        output_channels. It represents the output channel number of the RGB image after the Grayscale transformation.
        The output_channels should have values either 1 or 3 to represnt the output channel number.

        For exampel, for RGB image with shape (..., height, width, 3), after applying Grayscale transformation, 
        it will be as follows
        
            a. (..., height, width, 1) for output_channels = 1 , Or, 
            b. (..., height, width, 3) for output_channels = 3 .

        Here, ... notation represnts the batch size.


    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    to_grayscale = keras_cv.layers.preprocessing.Grayscale()
    augmented_images = to_grayscale(images)
    ```
    """

    def __init__(self, output_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels

    def _check_input_params(self, output_channels):
        if output_channels not in [1, 3]:
            raise ValueError(
                f"Received invalid argument output_channels. output_channels must be in 1 or 3. Got {output_channels}"
            ) 
        self.output_channels = output_channels

    def _rgb_to_grayscale(self, image):
        if self.output_channels == 1:
            return tf.image.rgb_to_grayscale(image)
        else:
            _grayscale = tf.image.rgb_to_grayscale(image)
            return tf.image.grayscale_to_rgb(_grayscale)
   
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

        return tf.cond(
            tf.cast(training, tf.bool), 
            lambda: self._rgb_to_grayscale(images), 
            lambda: images
            )
 

    def get_config(self):
        config = {
            "output_channels": self.output_channels,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape