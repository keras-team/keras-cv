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
from keras_cv.utils import preprocessing as preprocessing_utils


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomHue(VectorizedBaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the image hue is impacted.
            `factor=0.0` makes this layer perform a no-op operation, while a value of
            1.0 performs the most aggressive contrast adjustment available.  If a tuple
            is used, a `factor` is sampled between the two values for every image
            augmented.  If a single float is used, a value between `0.0` and the passed
            float is sampled.  In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.

    Usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_hue = keras_cv.layers.preprocessing.RandomHue()
    augmented_images = random_hue(images)
    ```
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing_utils.parse_factor(
            factor,
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        invert = self._random_generator.random_uniform((batch_size,), 0, 1, tf.float32)
        invert = tf.where(invert > 0.5, -tf.ones_like(invert), tf.ones_like(invert))
        # We must scale self.factor() to the range [-0.5, 0.5].  This is because the
        # tf.image operation performs rotation on the hue saturation value orientation.
        # This can be thought of as an angle in the range [-180, 180]
        return invert * self.factor(shape=(batch_size,)) * 0.5

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        adjust_factors = tf.cast(transformations, images.dtype)
        # broadcast
        adjust_factors = adjust_factors[..., tf.newaxis, tf.newaxis]

        images = tf.image.rgb_to_hsv(images)
        h_channel = images[..., 0] + adjust_factors
        h_channel = tf.where(h_channel > 1.0, h_channel - 1.0, h_channel)
        h_channel = tf.where(h_channel < 0.0, h_channel + 1.0, h_channel)
        images = tf.stack([h_channel, images[..., 1], images[..., 2]], axis=-1)
        images = tf.image.hsv_to_rgb(images)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(self, segmentation_masks, transformations, **kwargs):
        return segmentation_masks

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["factor"], dict):
            config["factor"] = tf.keras.utils.deserialize_keras_object(config["factor"])
        return cls(**config)
