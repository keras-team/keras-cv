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
from keras_cv.src.backend import keras
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils


@keras_cv_export("keras_cv.layers.RandomBrightness")
class RandomBrightness(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly adjusts brightness.

    This layer will randomly increase/reduce the brightness for the input RGB
    images.

    Note that different brightness adjustment factors
    will be applied to each the images in the batch.

    Args:
      factor: Float or a list/tuple of 2 floats between -1.0 and 1.0. The
        factor is used to determine the lower bound and upper bound of the
        brightness adjustment. A float value will be chosen randomly between
        the limits. When -1.0 is chosen, the output image will be black, and
        when 1.0 is chosen, the image will be fully white. When only one float
        is provided, eg, 0.2, then -0.2 will be used for lower bound and 0.2
        will be used for upper bound.
      value_range: Optional list/tuple of 2 floats for the lower and upper limit
        of the values of the input data, defaults to [0.0, 255.0]. Can be
        changed to e.g. [0.0, 1.0] if the image input has been scaled before
        this layer. The brightness adjustment will be scaled to this range, and
        the output values will be clipped to this range.
      seed: optional integer, for fixed RNG behavior.
    Inputs: 3D (HWC) or 4D (NHWC) tensor, with float or int dtype. Input pixel
      values can be of any range (e.g. `[0., 1.)` or `[0, 255]`)
    Output: 3D (HWC) or 4D (NHWC) tensor with brightness adjusted based on the
      `factor`. By default, the layer will output floats. The output value will
      be clipped to the range `[0, 255]`, the valid range of RGB colors, and
      rescaled based on the `value_range` if needed.

    Example:
    ```python
    (images, labels), _ = keras.datasets.cifar10.load_data()
    random_brightness = keras_cv.layers.preprocessing.RandomBrightness()
    augmented_images = random_brightness(images)
    ```
    """

    def __init__(self, factor, value_range=(0, 255), seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, float) or isinstance(factor, int):
            factor = (-factor, factor)
        self.factor = preprocessing_utils.parse_factor(
            factor, min_value=-1, max_value=1
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        rgb_delta_shape = (batch_size, 1, 1, 1)
        random_rgb_deltas = self.factor(shape=rgb_delta_shape)
        random_rgb_deltas = random_rgb_deltas * (
            self.value_range[1] - self.value_range[0]
        )
        return random_rgb_deltas

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        rank = images.shape.rank
        if rank != 4:
            raise ValueError(
                "Expected the input image to be rank 4. Got "
                f"inputs.shape = {images.shape}"
            )
        rgb_deltas = tf.cast(transformations, images.dtype)
        images += rgb_deltas
        return tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
        )

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
            "factor": self.factor,
            "value_range": self.value_range,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if isinstance(config["factor"], dict):
            config["factor"] = keras.utils.deserialize_keras_object(
                config["factor"]
            )
        return cls(**config)
