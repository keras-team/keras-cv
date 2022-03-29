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

from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomHue(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly adjusts the hue on given images.

    This layer will randomly increase/reduce the hue for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    The image hue is adjusted by converting the image(s) to HSV and rotating the
    hue channel (H) by delta. The image is then converted back to RGB.

    Args:
        factor: Either a tuple of two floats or a single float. `factor` controls the
            extent to which the image saturation is impacted. `factor` =
            `0.0`, `0.5` or `1.0` makes this layer perform a no-op operation.
            `factor=0.25` and `factor=0.75` makes the image to have fully opposite
            hue value. Values should be between `0.0` and `1.0`.
            If a tuple is used, a `factor` is sampled
            between the two values for every image augmented.  If a single float is
            used, a value between `0.0` and the passed float is sampled.
            In order to ensure the value is always the same, please pass a tuple with
            two identical floats: `(0.5, 0.5)`.
    """

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = preprocessing.parse_factor_value_range(
            factor, min_value=0.0, max_value=1.0
        )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        del image, label, bounding_box
        if self.factor[0] == self.factor[1]:
            return self.factor[0]
        return self._random_generator.random_uniform(
            shape=(), minval=self.factor[0], maxval=self.factor[1], dtype=tf.float32
        )

    def augment_image(self, image, transformation=None):
        # Convert the factor range from [0, 1] to [-1.0, 1.0].
        adjust_factor = transformation * 2.0 - 1.0
        return tf.image.adjust_hue(image, delta=adjust_factor)

    def get_config(self):
        config = {
            "factor": self.factor,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
