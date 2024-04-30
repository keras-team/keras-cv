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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)


@keras_cv_export("keras_cv.layers.ChannelShuffle")
class ChannelShuffle(VectorizedBaseImageAugmentationLayer):
    """Shuffle channels of an input image.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        groups: Number of groups to divide the input channels, defaults to 3.
        seed: Integer. Used to create a random seed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    channel_shuffle = ChannelShuffle(groups=3)
    augmented_images = channel_shuffle(images)
    ```
    """

    def __init__(self, groups=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.groups = groups
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # get batched shuffled indices
        # for example: batch_size=2; self.group=5
        # indices = [
        #     [0, 2, 3, 4, 1],
        #     [4, 1, 0, 2, 3]
        # ]
        indices_distribution = self._random_generator.uniform(
            (batch_size, self.groups)
        )
        indices = tf.argsort(indices_distribution, axis=-1)
        return indices

    def augment_ragged_image(self, image, transformation, **kwargs):
        # self.augment_images must have
        # 4D images (batch_size, height, width, channel)
        # 2D transformations (batch_size, groups)
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        batch_size = tf.shape(images)[0]
        height, width = images.shape[1], images.shape[2]
        num_channels = images.shape[3]
        indices = transformations

        # append batch indexes next to shuffled indices
        batch_indexs = tf.repeat(tf.range(batch_size), self.groups)
        batch_indexs = tf.reshape(batch_indexs, (batch_size, self.groups))
        indices = tf.stack([batch_indexs, indices], axis=-1)

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
        images = tf.transpose(images, perm=[0, 3, 1, 2, 4])
        images = tf.gather_nd(images, indices=indices)
        images = tf.transpose(images, perm=[0, 2, 3, 4, 1])
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
