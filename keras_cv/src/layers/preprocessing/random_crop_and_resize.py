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

from keras_cv.src import bounding_box
from keras_cv.src import core
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomCropAndResize")
class RandomCropAndResize(BaseImageAugmentationLayer):
    """Randomly crops a part of an image and resizes it to provided size.

    This implementation takes an intuitive approach, where we crop the images to
    a random height and width, and then resize them. To do this, we first sample
    a random value for area using `crop_area_factor` and a value for aspect
    ratio using `aspect_ratio_factor`. Further we get the new height and width
    by dividing and multiplying the old height and width by the random area
    respectively. We then sample offsets for height and width and clip them such
    that the cropped area does not exceed image boundaries. Finally, we do the
    actual cropping operation and resize the image to `target_size`.

    Args:
        target_size: A tuple of two integers used as the target size to
            ultimately crop images to.
        crop_area_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. The ratio of area of the cropped part to that
            of original image is sampled using this factor. Represents the lower
            and upper bounds for the area relative to the original image of the
            cropped image before resizing it to `target_size`. For
            self-supervised pretraining a common value for this parameter is
            `(0.08, 1.0)`. For fine tuning and classification a common value for
            this is `0.8, 1.0`.
        aspect_ratio_factor: A tuple of two floats, ConstantFactorSampler or
            UniformFactorSampler. Aspect ratio means the ratio of width to
            height of the cropped image. In the context of this layer, the
            aspect ratio sampled represents a value to distort the aspect ratio
            by. Represents the lower and upper bound for the aspect ratio of the
            cropped image before resizing it to `target_size`. For most tasks,
            this should be `(3/4, 4/3)`. To perform a no-op provide the value
            `(1.0, 1.0)`.
        interpolation: (Optional) A string specifying the sampling method for
            resizing, defaults to "bilinear".
        seed: (Optional) Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        target_size,
        crop_area_factor,
        aspect_ratio_factor,
        interpolation="bilinear",
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        self._check_class_arguments(
            target_size, crop_area_factor, aspect_ratio_factor
        )
        self.target_size = target_size
        self.aspect_ratio_factor = preprocessing.parse_factor(
            aspect_ratio_factor,
            min_value=0.0,
            max_value=None,
            param_name="aspect_ratio_factor",
            seed=seed,
        )
        self.crop_area_factor = preprocessing.parse_factor(
            crop_area_factor,
            max_value=1.0,
            param_name="crop_area_factor",
            seed=seed,
        )

        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.force_output_dense_images = True

    def get_random_transformation(
        self, image=None, label=None, bounding_box=None, **kwargs
    ):
        crop_area_factor = self.crop_area_factor()
        aspect_ratio = self.aspect_ratio_factor()

        new_height = tf.clip_by_value(
            tf.sqrt(crop_area_factor / aspect_ratio), 0.0, 1.0
        )  # to avoid unwanted/unintuitive effects
        new_width = tf.clip_by_value(
            tf.sqrt(crop_area_factor * aspect_ratio), 0.0, 1.0
        )

        height_offset = self._random_generator.uniform(
            (),
            minval=tf.minimum(0.0, 1.0 - new_height),
            maxval=tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )

        width_offset = self._random_generator.uniform(
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

    def compute_image_signature(self, images):
        return tf.TensorSpec(
            shape=(self.target_size[0], self.target_size[1], images.shape[-1]),
            dtype=self.compute_dtype,
        )

    def augment_image(self, image, transformation, **kwargs):
        return self._crop_and_resize(image, transformation)

    def augment_target(self, target, **kwargs):
        return target

    def _transform_bounding_boxes(bounding_boxes, transformation):
        bounding_boxes = bounding_boxes.copy()
        t_y1, t_x1, t_y2, t_x2 = transformation[0]
        t_dx = t_x2 - t_x1
        t_dy = t_y2 - t_y1
        x1, y1, x2, y2 = tf.split(
            bounding_boxes["boxes"], [1, 1, 1, 1], axis=-1
        )
        output = tf.concat(
            [
                (x1 - t_x1) / t_dx,
                (y1 - t_y1) / t_dy,
                (x2 - t_x1) / t_dx,
                (y2 - t_y1) / t_dy,
            ],
            axis=-1,
        )
        bounding_boxes["boxes"] = output
        return bounding_boxes

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCropAndResize()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCropAndResize(bounding_box_format='xyxy')`"
            )

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=image,
        )

        bounding_boxes = RandomCropAndResize._transform_bounding_boxes(
            bounding_boxes, transformation
        )

        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=image,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes

    def _resize(self, image, **kwargs):
        outputs = keras.preprocessing.image.smart_resize(
            image, self.target_size, **kwargs
        )
        # smart_resize will always output float32, so we need to re-cast.
        return tf.cast(outputs, self.compute_dtype)

    def _check_class_arguments(
        self, target_size, crop_area_factor, aspect_ratio_factor
    ):
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
            not isinstance(crop_area_factor, (tuple, list, core.FactorSampler))
            or isinstance(crop_area_factor, float)
            or isinstance(crop_area_factor, int)
        ):
            raise ValueError(
                "`crop_area_factor` must be tuple of two positive floats less "
                "than or equal to 1 or keras_cv.core.FactorSampler instance. "
                f"Received crop_area_factor={crop_area_factor}"
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

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return self._crop_and_resize(
            segmentation_mask, transformation, method="nearest"
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_size": self.target_size,
                "crop_area_factor": self.crop_area_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if isinstance(config["crop_area_factor"], dict):
            config["crop_area_factor"] = keras.utils.deserialize_keras_object(
                config["crop_area_factor"]
            )
        if isinstance(config["aspect_ratio_factor"], dict):
            config["aspect_ratio_factor"] = (
                keras.utils.deserialize_keras_object(
                    config["aspect_ratio_factor"]
                )
            )
        return cls(**config)

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
            self.target_size,  # output size
            method=method or self.interpolation,
        )

        return tf.squeeze(augmented_image, axis=0)
