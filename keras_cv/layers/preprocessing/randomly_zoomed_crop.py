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

from keras_cv import core
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomlyZoomedCrop(BaseImageAugmentationLayer):
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
            UniformFactorSampler. Represents the area relative to the original image
            of the cropped image before resizing it to `(height, width)`.
        aspect_ratio_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Aspect ratio means the ratio of width to
            height of the cropped image. In the context of this layer, the aspect ratio
            sampled represents a value to distort the aspect ratio by.
            Represents the lower and upper bound for the aspect ratio of the
            cropped image before resizing it to `(height, width)`.  For most tasks, this
            should be `(3/4, 4/3)`.  To perform a no-op provide the value `(1.0, 1.0)`.
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
        self.aspect_ratio_factor = preprocessing.parse_factor(
            aspect_ratio_factor,
            min_value=0.0,
            max_value=None,
            param_name="aspect_ratio_factor",
            seed=seed,
        )
        self.zoom_factor = preprocessing.parse_factor(
            zoom_factor,
            min_value=0.0,
            max_value=None,
            param_name="zoom_factor",
            seed=seed,
        )

        self._check_class_arguments(height, width, zoom_factor, aspect_ratio_factor)
        self.force_output_dense_images = True
        self.interpolation = interpolation
        self.seed = seed

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        zoom_factor = self.zoom_factor()
        aspect_ratio = self.aspect_ratio_factor()

        original_height = tf.cast(tf.shape(image)[-3], tf.float32)
        original_width = tf.cast(tf.shape(image)[-2], tf.float32)

        crop_size = (
            tf.round(self.height / zoom_factor),
            tf.round(self.width / zoom_factor),
        )

        new_height = crop_size[0] / tf.sqrt(aspect_ratio)

        new_width = crop_size[1] * tf.sqrt(aspect_ratio)

        height_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, original_height - new_height),
            maxval=tf.maximum(0.0, original_height - new_height),
            dtype=tf.float32,
        )

        width_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, original_width - new_width),
            maxval=tf.maximum(0.0, original_width - new_width),
            dtype=tf.float32,
        )

        new_height = new_height / original_height
        new_width = new_width / original_width

        height_offset = height_offset / original_height
        width_offset = width_offset / original_width

        return (new_height, new_width, height_offset, width_offset)

    def call(self, inputs, training=True):
        if training:
            return super().call(inputs, training)
        else:
            inputs = self._ensure_inputs_are_compute_dtype(inputs)
            inputs, meta_data = self._format_inputs(inputs)
            output = inputs
            # self._resize() returns valid results for both batched and
            # unbatched
            output["images"] = self._resize(inputs["images"])

            return self._format_output(output, meta_data)

    def augment_image(self, image, transformation, **kwargs):
        image_shape = tf.shape(image)

        height = tf.cast(image_shape[-3], tf.float32)
        width = tf.cast(image_shape[-2], tf.float32)

        image = tf.expand_dims(image, axis=0)
        new_height, new_width, height_offset, width_offset = transformation

        transform = RandomlyZoomedCrop._format_transform(
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
            output_shape=(self.height, self.width),
            interpolation=self.interpolation,
            fill_mode="reflect",
        )

        return tf.squeeze(image, axis=0)

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def _resize(self, image):
        outputs = tf.keras.preprocessing.image.smart_resize(
            image, (self.height, self.width)
        )
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def _check_class_arguments(self, height, width, zoom_factor, aspect_ratio_factor):
        if not isinstance(height, int):
            raise ValueError("`height` must be an integer. Received height={height}")

        if not isinstance(width, int):
            raise ValueError("`width` must be an integer. Received width={width}")

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
            not isinstance(aspect_ratio_factor, (tuple, list, core.FactorSampler))
            or isinstance(aspect_ratio_factor, float)
            or isinstance(aspect_ratio_factor, int)
        ):
            raise ValueError(
                "`aspect_ratio_factor` must be tuple of two positive floats or "
                "keras_cv.core.FactorSampler instance. Received "
                f"aspect_ratio_factor={aspect_ratio_factor}"
            )

    def augment_target(self, augment_target, **kwargs):
        return augment_target

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

    def _crop_and_resize(self, image, transformation, method=None):
        image = tf.expand_dims(image, axis=0)
        boxes = transformation

        # See bit.ly/tf_crop_resize for more details
        augmented_image = tf.image.crop_and_resize(
            image,  # image shape: [B, H, W, C]
            boxes,  # boxes: (1, 4) in this case; represents area
            # to be cropped from the original image
            [0],  # box_indices: maps boxes to images along batch axis
            # [0] since there is only one image
            (self.height, self.width),  # output size
            method=method or self.interpolation,
        )

        return tf.squeeze(augmented_image, axis=0)
