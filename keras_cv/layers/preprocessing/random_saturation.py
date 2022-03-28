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
import numpy as np
import tensorflow as tf

from keras_cv.utils import preprocessing


class RandomSaturation(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly adjusts the saturation on given images.

    This layer will randomly increase/reduce the saturation for the input RGB
    images. At inference time, the output will be identical to the input.
    Call the layer with `training=True` to adjust the brightness of the input.

    Args:
        factor: Either a tuple of two floats or a single float. `factor` controls the
            extent to which the image saturation is impacted. `factor=1.0` makes
            this layer perform a no-op operation. `factor=0.0` makes the image to be
            fully grayscale. Any value larger than 1.0 will increase the saturation
            of the image.

            Values should be between `0.0` and +inf. If a tuple is used, a `factor`
            is sampled between the two values for every image augmented.  If a single
            float is used, a value between `0.0` and the passed float is sampled.
            In order to ensure the value is always the same, please pass a tuple with
            two identical floats: `(0.5, 0.5)`.
    """

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = preprocessing.parse_factor_value_range(
            factor, min_value=0.0, max_value=np.inf
        )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        del image, label, bounding_box
        if self.factor[0] == self.factor[1]:
            return self.factor[0]
        return self._random_generator.random_uniform(
            shape=(), minval=self.factor[0], maxval=self.factor[1], dtype=tf.float32
        )

    def augment_image(self, image, transformation=None):
        adjust_factor = transformation
        return tf.image.adjust_saturation(image, saturation_factor=adjust_factor)

    def get_config(self):
        # TODO(scottzhu): Add tf.keras.utils.register_keras_serializable to all the
        # KPLs for serial/deserialization
        config = {
            "factor": self.factor,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
