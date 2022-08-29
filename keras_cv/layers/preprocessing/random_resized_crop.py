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
class RandomResizedCrop(BaseImageAugmentationLayer):
    """Randomly crops a part of an image and resizes it to provided size.

    This implementation takes a distortion-oriented approach, which means the
    amount of distortion in the image is proportional to the `zoom_factor`
    argument. To do this, we first sample a random value for `zoom_factor` and
    `aspect_ratio_factor`. Further we deduce a `crop_size` which abides by the
    calculated aspect ratio. Finally we do the actual cropping operation and
    resize the image to `target_size`.

    Args:
        target_size: A tuple of two integers used as the target size to ultimately crop
            images to.
        zoom_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Represents the area relative to the original image
            of the cropped image before resizing it to `target_size`.
        aspect_ratio_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Aspect ratio means the ratio of width to
            height of the cropped image. In the context of this layer, the aspect ratio
            sampled represents a value to distort the aspect ratio by.
            Represents the lower and upper bound for the aspect ratio of the
            cropped image before resizing it to `target_size`.  For most tasks, this
            should be `(3/4, 4/3)`.  To perform a no-op provide the value `(1.0, 1.0)`.
        interpolation: (Optional) A string specifying the sampling method for
            resizing. Defaults to "bilinear".
        seed: (Optional) Used to create a random seed. Defaults to None.
    """

    def __init__(
        self,
        target_size,
        zoom_factor,
        aspect_ratio_factor,
        interpolation="bilinear",
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self._check_class_arguments(target_size, zoom_factor, aspect_ratio_factor)

        self.target_size = target_size
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

        self.interpolation = interpolation
        self.seed = seed

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        zoom_factor = self.zoom_factor()
        aspect_ratio = self.aspect_ratio_factor()

        if image.shape.rank == 3:
            original_height = image.shape[0]
            original_width = image.shape[1]
        elif image.shape.rank == 4:
            original_height = image.shape[1]
            original_width = image.shape[2]

        crop_size = (
            tf.round(self.target_size[0] / zoom_factor),
            tf.round(self.target_size[1] / zoom_factor),
        )

        new_height = tf.clip_by_value(
            crop_size[0] / tf.sqrt(aspect_ratio),
            0.0,
            tf.cast(original_height, tf.float32),
        )  # to avoid unwanted/unintuitive effects
        new_width = tf.clip_by_value(
            crop_size[1] * tf.sqrt(aspect_ratio),
            0.0,
            tf.cast(original_width, tf.float32),
        )

        height_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, 1.0 - new_height),
            maxval=tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )

        width_offset = self._random_generator.random_uniform(
            (),
            minval=tf.minimum(0.0, 1.0 - new_width),
            maxval=tf.maximum(0.0, 1.0 - new_width),
            dtype=tf.float32,
        )

        y1 = height_offset
        y2 = height_offset + new_height
        x1 = width_offset
        x2 = width_offset + new_width

        return [[y1, x1, y2, x2]]

    def call(self, inputs, training=True):

        if training:
            return super().call(inputs, training)
        else:
            inputs = self._ensure_inputs_are_compute_dtype(inputs)
            inputs, is_dict, use_targets = self._format_inputs(inputs)
            output = inputs
            # self._resize() returns valid results for both batched and
            # unbatched
            output["images"] = self._resize(inputs["images"])
            return self._format_output(output, is_dict, use_targets)

    def augment_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        boxes = transformation

        # See bit.ly/tf_crop_resize for more details
        augmented_image = tf.image.crop_and_resize(
            image,  # image shape: [B, H, W, C]
            boxes,  # boxes: (1, 4) in this case; represents area
            # to be cropped from the original image
            [0],  # box_indices: maps boxes to images along batch axis
            # [0] since there is only one image
            self.target_size,  # output size
        )

        return tf.squeeze(augmented_image, axis=0)

    def _resize(self, image):
        outputs = tf.keras.preprocessing.image.smart_resize(image, self.target_size)
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def _check_class_arguments(self, target_size, zoom_factor, aspect_ratio_factor):
        if (
            not isinstance(target_size, (tuple, list))
            or len(target_size) != 2
            or not isinstance(target_size[0], int)
            or not isinstance(target_size[1], int)
            or isinstance(target_size, int)
        ):
            raise ValueError(
                "`target_size` must be tuple of two integers. "
                f"Received target_size={target_size}"
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
                "target_size": self.target_size,
                "zoom_factor": self.zoom_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "seed": self.seed,
            }
        )
        return config
