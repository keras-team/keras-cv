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
import warnings

import tensorflow as tf

from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomShear(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly shears an image.

    Args:
        x: float, 2 element tuple, or `None`.  For each augmented image a value is
            sampled from the provided range.  If a float is passed, the range is
            interpreted as `(0, x)`.  Values represent a percentage of the image
            to shear over.  For example, 0.3 shears pixels up to 30% of the way
            across the image.  All provided values should be positive.  If
            `None` is passed, no shear occurs on the X axis.  Defaults to `None`.
        y: float, 2 element tuple, or `None`.  For each augmented image a value is
            sampled from the provided range.  If a float is passed, the range is
            interpreted as `(0, y)`.  Values represent a percentage of the image
            to shear over.  For example, 0.3 shears pixels up to 30% of the way
            across the image.  All provided values should be positive.  If
            `None` is passed, no shear occurs on the Y axis.  Defaults to `None`.
        interpolation: interpolation method used in the `ImageProjectiveTransformV3` op.
             Supported values are `"nearest"` and `"bilinear"`.
             Defaults to `"bilinear"`.
        fill_mode: fill_mode in the `ImageProjectiveTransformV3` op.
             Supported values are `"reflect"`, `"wrap"`, `"constant"`, and `"nearest"`.
             Defaults to `"reflect"`.
        fill_value: fill_value in the `ImageProjectiveTransformV3` op.
             A `Tensor` of type `float32`. The value to be filled when fill_mode is
             constant".  Defaults to `0.0`.
    """

    def __init__(
        self,
        x=None,
        y=None,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(x, float):
            x = (0, x)
        if isinstance(y, float):
            y = (0, y)
        if x is None and y is None:
            warnings.warn(
                "RandomShear received both `x=None` and `y=None`.  As a "
                "result, the layer will perform no augmentation."
            )
        self.x = x
        self.y = y
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        x = self._get_shear_amount(self.x)
        y = self._get_shear_amount(self.y)
        return (x, y)

    def _get_shear_amount(self, constraint):
        if constraint is None:
            return None

        negate = self._random_generator.random_uniform((), 0, 1, dtype=tf.float32) > 0.5
        negate = tf.cond(negate, lambda: -1.0, lambda: 1.0)

        return negate * self._random_generator.random_uniform(
            (), constraint[0], constraint[1]
        )

    def augment_image(self, image, transformation=None):
        image = tf.expand_dims(image, axis=0)

        x, y = transformation

        if x is not None:
            transform_x = RandomShear._format_transform(
                [1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
            image = preprocessing.transform(
                images=image,
                transforms=transform_x,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        if y is not None:
            transform_y = RandomShear._format_transform(
                [1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]
            )
            image = preprocessing.transform(
                images=image,
                transforms=transform_y,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        return tf.squeeze(image, axis=0)

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x": self.x,
                "y": self.y,
                "interpolation": self.interpolation,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
            }
        )
        return config
