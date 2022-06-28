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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class RandomContrast(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly adjusts contrast during training.

    This layer will randomly adjust the contrast of an image or images by a random
    factor. Contrast is adjusted independently for each channel of each image
    during training.

    For each channel, this layer computes the mean of the image pixels in the
    channel and then adjusts each component `x` of each pixel to
    `(x - mean) * contrast_factor + mean`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    in integer or floating point dtype. By default, the layer will output floats.
    The output value will be clipped to the range `[0, 255]`, the valid
    range of RGB colors.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Arguments:
      factor: a positive float represented as fraction of value, or a tuple of
        size 2 representing lower and upper bound. When represented as a single
        float, lower = upper. The contrast factor will be randomly picked between
        `[1.0 - lower, 1.0 + upper]`. For any pixel x in the channel, the output
        will be `(x - mean) * factor + mean` where `mean` is the mean value of the
        channel.
      seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.base = tf.keras.layers.RandomContrast(factor=factor, seed=seed, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(self.base.get_config())
        return config

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        return self.base.get_random_transformation(
            image=image, label=label, bounding_box=bounding_box
        )

    def augment_image(self, image, transformation=None, **kwargs):
        return self.base.augment_image(image=image, transformation=transformation)

    def augment_label(self, labels, transformation=None, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_keypoints(self, keypoints, **kwargs):
        return keypoints
