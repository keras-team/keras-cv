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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomShear")
class RandomShear(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly shears images.

    This layer will apply random shearings to each image, filling empty space
    according to `fill_mode`.

    Input pixel values can be of any range and any data type.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        x_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, x_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the X axis. Defaults to `None`.
        y_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is
            sampled from the provided range. If a float is passed, the range is
            interpreted as `(0, y_factor)`. Values represent a percentage of the
            image to shear over. For example, 0.3 shears pixels up to 30% of the
            way across the image. All provided values should be positive. If
            `None` is passed, no shear occurs on the Y axis. Defaults to `None`.
        interpolation: interpolation method used in the
            `ImageProjectiveTransformV3` op. Supported values are `"nearest"`
            and `"bilinear"`, defaults to `"bilinear"`.
        fill_mode: fill_mode in the `ImageProjectiveTransformV3` op. Supported
            values are `"reflect"`, `"wrap"`, `"constant"`, and `"nearest"`.
            Defaults to `"reflect"`.
        fill_value: fill_value in the `ImageProjectiveTransformV3` op. A
            `Tensor` of type `float32`. The value to be filled when fill_mode is
            constant". Defaults to `0.0`.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer to
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if x_factor is not None:
            self.x_factor = preprocessing.parse_factor(
                x_factor, max_value=None, param_name="x_factor", seed=seed
            )
        else:
            self.x_factor = x_factor
        if y_factor is not None:
            self.y_factor = preprocessing.parse_factor(
                y_factor, max_value=None, param_name="y_factor", seed=seed
            )
        else:
            self.y_factor = y_factor
        if x_factor is None and y_factor is None:
            warnings.warn(
                "RandomShear received both `x_factor=None` and `y_factor=None`."
                " As a result, the layer will perform no augmentation."
            )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation_batch(self, batch_size, **kwargs):
        transformations = {"shear_x": None, "shear_y": None}
        if self.x_factor is not None:
            invert = preprocessing.batch_random_inversion(
                self._random_generator, batch_size
            )
            transformations["shear_x"] = (
                self.x_factor(shape=(batch_size, 1)) * invert
            )

        if self.y_factor is not None:
            invert = preprocessing.batch_random_inversion(
                self._random_generator, batch_size
            )
            transformations["shear_y"] = (
                self.y_factor(shape=(batch_size, 1)) * invert
            )

        return transformations

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        new_transformation = {"shear_x": None, "shear_y": None}
        shear_x = transformation["shear_x"]
        if shear_x is not None:
            new_transformation["shear_x"] = tf.expand_dims(shear_x, axis=0)

        shear_y = transformation["shear_y"]
        if shear_y is not None:
            new_transformation["shear_y"] = tf.expand_dims(shear_y, axis=0)

        output = self.augment_images(images, new_transformation)
        return tf.squeeze(output, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        x, y = transformations["shear_x"], transformations["shear_y"]

        if x is not None:
            transforms_x = self._build_shear_x_transform_matrix(x)
            images = preprocessing.transform(
                images=images,
                transforms=transforms_x,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        if y is not None:
            transforms_y = self._build_shear_y_transform_matrix(y)
            images = preprocessing.transform(
                images=images,
                transforms=transforms_y,
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        return images

    @staticmethod
    def _build_shear_x_transform_matrix(shear_x):
        """Build transform matrix for horizontal shear.

        The transform matrix looks like:
        (1, x, 0)
        (0, 1, 0)
        (0, 0, 1)
        where the last entry is implicit.

        We flatten the matrix to `[1.0, x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]` for
        use with ImageProjectiveTransformV3.
        """
        batch_size = tf.shape(shear_x)[0]
        return tf.concat(
            values=[
                tf.ones((batch_size, 1), tf.float32),
                shear_x,
                tf.zeros((batch_size, 2), tf.float32),
                tf.ones((batch_size, 1), tf.float32),
                tf.zeros((batch_size, 3), tf.float32),
            ],
            axis=1,
        )

    @staticmethod
    def _build_shear_y_transform_matrix(shear_y):
        """Build transform matrix for vertical shear.

        The transform matrix looks like:
        (1, 0, 0)
        (y, 1, 0)
        (0, 0, 1)
        where the last entry is implicit.

        We flatten the matrix to `[1.0, 0.0, 0.0, y, 1.0, 0.0, 0.0, 0.0]` for
        use ImageProjectiveTransformV3.
        """
        batch_size = tf.shape(shear_y)[0]
        return tf.concat(
            values=[
                tf.ones((batch_size, 1), tf.float32),
                tf.zeros((batch_size, 2), tf.float32),
                shear_y,
                tf.ones((batch_size, 1), tf.float32),
                tf.zeros((batch_size, 3), tf.float32),
            ],
            axis=1,
        )

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        x, y = transformations["shear_x"], transformations["shear_y"]

        if x is not None:
            transforms_x = self._build_shear_x_transform_matrix(x)
            segmentation_masks = preprocessing.transform(
                images=segmentation_masks,
                transforms=transforms_x,
                interpolation="nearest",
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        if y is not None:
            transforms_y = self._build_shear_y_transform_matrix(y)
            segmentation_masks = preprocessing.transform(
                images=segmentation_masks,
                transforms=transforms_y,
                interpolation="nearest",
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
            )

        return segmentation_masks

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        """Augments bounding boxes after a shear operations.

        The algorithm to update (x,y) point coordinates after shearing, tells us
        to matrix multiply them with inverted transform matrix. This is:
        ```
        # for shear x              # for shear_y
        (1.0, -shear_x) (x)        (1.0,      0.0) (x)
        (0.0, 1.0     ) (y)        (-shear_y, 1.0) (y)
        ```
        We can simplify this equation: any new coordinate can be calculated by
        `x = x - (shear_x * y)` and `(y = y - (shear_y * x)`

        Notice that each coordinate has to be calculated twice, e.g. `x1` will
        be affected differently by y1 (top) and y2 (bottom). Therefore, we
        calculate both `x1_top` and `x1_bottom` and choose the final x1
        depending on the sign of the used shear value.
        """
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomShear()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomShear(bounding_box_format='xyxy')`"
            )

        # Edge case: boxes is a tf.RaggedTensor
        if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
            bounding_boxes = bounding_box.to_dense(
                bounding_boxes, default_value=0
            )

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=images,
            dtype=self.compute_dtype,
        )

        shear_x_amount = transformations["shear_x"]
        shear_y_amount = transformations["shear_y"]
        x1, y1, x2, y2 = tf.split(bounding_boxes["boxes"], 4, axis=-1)

        # Squeeze redundant extra dimension as it messes multiplication
        # [num_batches, num_boxes, 1] -> [num_batches, num_boxes]
        x1 = tf.squeeze(x1, axis=-1)
        y1 = tf.squeeze(y1, axis=-1)
        x2 = tf.squeeze(x2, axis=-1)
        y2 = tf.squeeze(y2, axis=-1)

        # Apply horizontal shear
        if shear_x_amount is not None:
            x1_top = x1 - (shear_x_amount * y1)
            x1_bottom = x1 - (shear_x_amount * y2)
            x1 = tf.where(shear_x_amount < 0, x1_top, x1_bottom)

            x2_top = x2 - (shear_x_amount * y1)
            x2_bottom = x2 - (shear_x_amount * y2)
            x2 = tf.where(shear_x_amount < 0, x2_bottom, x2_top)

        # Apply vertical shear
        if shear_y_amount is not None:
            y1_left = y1 - (shear_y_amount * x1)
            y1_right = y1 - (shear_y_amount * x2)
            y1 = tf.where(shear_y_amount > 0, y1_right, y1_left)

            y2_left = y2 - (shear_y_amount * x1)
            y2_right = y2 - (shear_y_amount * x2)
            y2 = tf.where(shear_y_amount > 0, y2_left, y2_right)

        # Join the results:
        boxes = tf.concat(
            [
                # Add dummy last axis for concat:
                # (num_batches, num_boxes) -> (num_batches, num_boxes, 1)
                x1[..., tf.newaxis],
                y1[..., tf.newaxis],
                x2[..., tf.newaxis],
                y2[..., tf.newaxis],
            ],
            axis=-1,
        )

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, images=images, bounding_box_format="rel_xyxy"
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            images=images,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "x_factor": self.x_factor,
                "y_factor": self.y_factor,
                "interpolation": self.interpolation,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
