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
from keras import backend
from tensorflow import keras

from keras_cv import core
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    IMAGES,
)
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing as preprocessing_utils

H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_cv")
class RandomlyZoomedCrop(VectorizedBaseImageAugmentationLayer):
    """Randomly crops a part of an image and zooms it by a provided amount size.

    This implementation takes a distortion-oriented approach, which means the
    amount of distortion in the image is proportional to the `zoom_factor`
    argument. To do this, we first sample a random value for `zoom_factor` and
    `aspect_ratio_factor`. Further we deduce a `crop_size` which abides by the
    calculated aspect ratio. Finally we do the actual cropping operation and
    resize the image to `(height, width)`.

    Args:
        height: The height of the output shape.
        width: The width of the output shape.
        zoom_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Represents the area relative to the original
            image of the cropped image before resizing it to `(height, width)`.
        aspect_ratio_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Aspect ratio means the ratio of width to
            height of the cropped image. In the context of this layer, the
            aspect ratio sampled represents a value to distort the aspect ratio
            by.
            Represents the lower and upper bound for the aspect ratio of the
            cropped image before resizing it to `(height, width)`.  For most
            tasks, this should be `(3/4, 4/3)`.  To perform a no-op provide the
            value `(1.0, 1.0)`.
        interpolation: (Optional) A string specifying the sampling method for
            resizing. Defaults to "bilinear".
        seed: (Optional) Used to create a random seed. Defaults to None.
    """

    def __init__(
        self,
        height,
        width,
        zoom_factor,
        aspect_ratio_factor,
        interpolation="bilinear",
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.height = height
        self.width = width
        self.aspect_ratio_factor = preprocessing_utils.parse_factor(
            aspect_ratio_factor,
            min_value=0.0,
            max_value=None,
            param_name="aspect_ratio_factor",
            seed=seed,
        )
        self.zoom_factor = preprocessing_utils.parse_factor(
            zoom_factor,
            min_value=0.0,
            max_value=None,
            param_name="zoom_factor",
            seed=seed,
        )

        self._check_class_arguments(
            height, width, zoom_factor, aspect_ratio_factor
        )
        self.force_output_dense_images = True
        self.interpolation = interpolation
        self.seed = seed

    def _check_class_arguments(
        self, height, width, zoom_factor, aspect_ratio_factor
    ):
        if not isinstance(height, int):
            raise ValueError(
                f"`height` must be an integer. Received height={height}"
            )

        if not isinstance(width, int):
            raise ValueError(
                f"`width` must be an integer. Received width={width}"
            )

        if (
            not isinstance(zoom_factor, (tuple, list, core.FactorSampler))
            or isinstance(zoom_factor, float)
            or isinstance(zoom_factor, int)
        ):
            raise ValueError(
                "`zoom_factor` must be tuple of two positive floats"
                " or keras_cv.core.FactorSampler instance. Received "
                f"zoom_factor={zoom_factor}"
            )

        if (
            not isinstance(
                aspect_ratio_factor, (tuple, list, core.FactorSampler)
            )
            or isinstance(aspect_ratio_factor, float)
            or isinstance(aspect_ratio_factor, int)
        ):
            raise ValueError(
                "`aspect_ratio_factor` must be tuple of two positive floats or "
                "keras_cv.core.FactorSampler instance. Received "
                f"aspect_ratio_factor={aspect_ratio_factor}"
            )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        zoom_factors = self.zoom_factor(shape=(batch_size, 1))
        aspect_ratios = self.aspect_ratio_factor(shape=(batch_size,))

        heights = tf.cast(
            tf.tile([self.height], multiples=(batch_size,)), tf.float32
        )
        widths = tf.cast(
            tf.tile([self.width], multiples=(batch_size,)), tf.float32
        )
        crop_size = (
            tf.round(heights / zoom_factors),
            tf.round(widths / zoom_factors),
        )

        new_heights = crop_size[0] / tf.sqrt(aspect_ratios)
        new_widths = crop_size[1] * tf.sqrt(aspect_ratios)

        height_offsets = self._random_generator.random_uniform(
            shape=(batch_size, 1), maxval=1.0, dtype=tf.float32
        )
        width_offsets = self._random_generator.random_uniform(
            shape=(batch_size, 1), maxval=1.0, dtype=tf.float32
        )

        return {
            "new_heights": new_heights,
            "new_widths": new_widths,
            "height_offsets": height_offsets,
            "width_offsets": width_offsets,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        new_heights = transformation["new_heights"]
        new_widths = transformation["new_widths"]
        height_offsets = transformation["height_offsets"]
        width_offsets = transformation["width_offsets"]
        transformation = {
            "new_heights": tf.expand_dims(new_heights, axis=0),
            "new_widths": tf.expand_dims(new_widths, axis=0),
            "height_offsets": tf.expand_dims(height_offsets, axis=0),
            "width_offsets": tf.expand_dims(width_offsets, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        image_shape = tf.shape(images)
        image_height = tf.cast(image_shape[H_AXIS], tf.float32)
        image_width = tf.cast(image_shape[W_AXIS], tf.float32)

        new_widths = transformations["new_widths"]
        new_heights = transformations["new_heights"]
        width_offsets = transformations["width_offsets"]
        height_offsets = transformations["height_offsets"]

        zooms = tf.concat(
            [new_widths / image_width, new_heights / image_height], axis=1
        )
        offsets = tf.concat([width_offsets, height_offsets], axis=1)
        transforms = self.get_zoomed_crop_matrix(
            zooms, offsets, image_height, image_width
        )

        images = preprocessing_utils.transform(
            images=images,
            transforms=transforms,
            output_shape=(self.height, self.width),
            interpolation=self.interpolation,
            fill_mode="reflect",
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def get_zoomed_crop_matrix(
        self, zooms, offsets, image_height, image_width, name=None
    ):
        """Returns projective transform(s) for the given zoom(s) and offset(s).

        Args:
        zooms: A matrix of 2-element lists representing `[zx, zy]` to zoom for
            each image (for a batch of images).
        offsets: A matrix of 2-element lists representing `[ox, oy]` to offset
            for each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

        Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v3`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
        """
        with backend.name_scope(name or "zoomed_crop_matrix"):
            batch_size = tf.shape(zooms)[0]
            new_widths = zooms[:, 0] * image_width
            new_heights = zooms[:, 1] * image_height
            image_heights = tf.cast(
                tf.tile([image_height], multiples=(batch_size,)), tf.float32
            )
            image_widths = tf.cast(
                tf.tile([image_width], multiples=(batch_size,)), tf.float32
            )
            width_offsets = offsets[:, 0] * (image_widths - new_widths)
            height_offsets = offsets[:, 1] * (image_heights - new_heights)
            return tf.concat(
                values=[
                    zooms[:, 0, tf.newaxis],
                    tf.zeros((batch_size, 1), tf.float32),
                    width_offsets[:, tf.newaxis],
                    tf.zeros((batch_size, 1), tf.float32),
                    zooms[:, 1, tf.newaxis],
                    height_offsets[:, tf.newaxis],
                    tf.zeros((batch_size, 2), tf.float32),
                ],
                axis=1,
            )

    def _resize(self, images, **kwargs):
        resizing_layer = keras.layers.Resizing(
            self.height, self.width, **kwargs
        )
        outputs = resizing_layer(images)
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "zoom_factor": self.zoom_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config["zoom_factor"], dict):
            config["zoom_factor"] = keras.utils.deserialize_keras_object(
                config["zoom_factor"]
            )
        if isinstance(config["aspect_ratio_factor"], dict):
            config[
                "aspect_ratio_factor"
            ] = keras.utils.deserialize_keras_object(
                config["aspect_ratio_factor"]
            )
        return cls(**config)
