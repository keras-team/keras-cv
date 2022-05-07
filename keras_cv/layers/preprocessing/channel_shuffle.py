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

import keras_cv


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class ChannelShuffle(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
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
        inputs: Tensor representing images of shape
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
        super().__init__(seed=seed, **kwargs)
        self.groups = groups
        self.seed = seed

    def augment_image(self, image, transformation=None):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        num_channels = image.shape[2]

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )

        channels_per_group = num_channels // self.groups

        rand_uniform = keras_cv.UniformFactorSampler(lower=0, upper=1, seed=self.seed)
        rand_indices = tf.argsort(rand_uniform(shape=[self.groups]))

        image = tf.reshape(image, [height, width, channels_per_group, self.groups])
        image = tf.gather(image, rand_indices, axis=-1)
        image = tf.reshape(image, [height, width, num_channels])

        return image

    def augment_label(self, label, transformation=None):
        return label

    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups, "seed": self.seed})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
