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

import tensorflow as tf

from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class ChannelShuffle(VectorizedBaseImageAugmentationLayer):
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

    def augment_ragged_image(self, image, transformation, **kwargs):
        # augment_images needs 4D (batch_size, height, width, channel)
        image = tf.expand_dims(image, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        batch_size = images.shape[0]
        height, width = images.shape[1], images.shape[2]
        num_channels = images.shape[3]

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )

        channels_per_group = num_channels // self.groups
        images = tf.reshape(
            images, [batch_size, height, width, self.groups, channels_per_group]
        )
        images = tf.transpose(images, perm=[3, 0, 1, 2, 4])
        images = tf.random.shuffle(images, seed=self.seed)
        images = tf.transpose(images, perm=[1, 2, 3, 4, 0])
        images = tf.reshape(images, [batch_size, height, width, num_channels])

        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def get_config(self):
        config = {
            "groups": self.groups,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
