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
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomSaturation(BaseImageAugmentationLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    Args:
        factor: A tuple of two floats, a single float or `keras_cv.FactorSampler`.
            `factor` controls the extent to which the image saturation is impacted.
            `factor=0.5` makes this layer perform a no-op operation. `factor=0.0` makes
            the image to be fully grayscale. `factor=1.0` makes the image to be fully
            saturated.
            Values should be between `0.0` and `1.0`. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.  If a single
            float is used, a value between `0.0` and the passed float is sampled.
            In order to ensure the value is always the same, please pass a tuple with
            two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.factor = preprocessing.parse_factor(
            factor,
            min_value=0.0,
            max_value=1.0,
        )
        self.seed = seed

    def get_random_transformation(self, **kwargs):
        return self.factor()

    def augment_image(self, image, transformation=None, **kwargs):
        # Convert the factor range from [0, 1] to [0, +inf]. Note that the
        # tf.image.adjust_saturation is trying to apply the following math formula
        # `output_saturation = input_saturation * factor`. We use the following
        # method to the do the mapping.
        # `y = x / (1 - x)`.
        # This will ensure:
        #   y = +inf when x = 1 (full saturation)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (full gray scale)

        # Convert the transformation to tensor in case it is a float. When
        # transformation is 1.0, then it will result in to divide by zero error, but
        # it will be handled correctly when it is a one tensor.
        transformation = tf.convert_to_tensor(transformation)
        adjust_factor = transformation / (1 - transformation)
        return tf.image.adjust_saturation(image, saturation_factor=adjust_factor)

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = {
            "factor": self.factor,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
