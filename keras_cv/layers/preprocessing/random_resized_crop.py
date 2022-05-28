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
class RandomResizedCrop(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Randomly crops an image and resizes it to its original resolution.
    Args:
        TODO
        interpolation: interpolation method used in the `ImageProjectiveTransformV3` op.
             Supported values are `"nearest"` and `"bilinear"`.
             Defaults to `"bilinear"`.
        fill_mode: fill_mode in the `ImageProjectiveTransformV3` op.
             Supported values are `"reflect"`, `"wrap"`, `"constant"`, and `"nearest"`.
             Defaults to `"reflect"`.
        fill_value: fill_value in the `ImageProjectiveTransformV3` op.
             A `Tensor` of type `float32`. The value to be filled when fill_mode is
             constant".  Defaults to `0.0`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        area_factor=0.0,
        aspect_ratio_factor=0.0,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.area_factor = preprocessing.parse_factor(
            area_factor, param_name="area_factor", seed=seed
        )
        self.aspect_ratio_factor = preprocessing.parse_factor(
            aspect_ratio_factor, param_name="aspect_ratio_factor", seed=seed
        )
        if area_factor == 0.0 and aspect_ratio_factor == 0.0:
            warnings.warn(
                "RandomResizedCrop received both `area_factor=0.0` and "
                "`aspect_ratio_factor=0.0`. As a result, the layer will perform no "
                "augmentation."
            )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        # random area and aspect ratio
        random_area = (
            1.0
            + preprocessing.random_inversion(self._random_generator)
            * self.area_factor()
        )
        random_aspect_ratio = (
            1.0
            + preprocessing.random_inversion(self._random_generator)
            * self.aspect_ratio_factor()
        )

        # corresponding height and width (1 = original height/width)
        new_height = tf.sqrt(random_area / random_aspect_ratio)
        new_width = tf.sqrt(random_area * random_aspect_ratio)

        # random offsets for the crop, inside or outside the image
        height_offset = self._random_generator.random_uniform(
            (),
            tf.minimum(0.0, 1.0 - new_height),
            tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )
        width_offset = self._random_generator.random_uniform(
            (),
            tf.minimum(0.0, 1.0 - new_width),
            tf.maximum(0.0, 1.0 - new_width),
            dtype=tf.float32,
        )

        return (new_height, new_width, height_offset, width_offset)

    def augment_image(self, image, transformation=None):
        height, width, _ = image.shape
        image = tf.expand_dims(image, axis=0)

        new_height, new_width, height_offset, width_offset = transformation

        transform = RandomResizedCrop._format_transform(
            [
                new_width,
                0.0,
                width_offset * width,
                0.0,
                new_height,
                height_offset * height,
                0.0,
                0.0,
            ]
        )
        image = preprocessing.transform(
            images=image,
            transforms=transform,
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )

        return tf.squeeze(image, axis=0)

    def augment_label(self, label, transformation=None):
        return label

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "area_factor": self.area_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "seed": self.seed,
            }
        )
        return config
