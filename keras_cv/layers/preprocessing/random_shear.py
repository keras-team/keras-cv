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

import keras_cv
from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomShear(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly shears images during training.
    This layer will apply random shearings to each image, filling empty space
    according to `fill_mode`.
    By default, random shears are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    shear at inference time, set `training` to True when calling the layer.
    Input pixel values can be of any range and any data type.
    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        x_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is sampled
            from the provided range. If a float is passed, the range is interpreted as
            `(0, x_factor)`.  Values represent a percentage of the image to shear over.
             For example, 0.3 shears pixels up to 30% of the way across the image.
             All provided values should be positive.  If `None` is passed, no shear
             occurs on the X axis.
             Defaults to `None`.
        y_factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. For each augmented image a value is sampled
            from the provided range. If a float is passed, the range is interpreted as
            `(0, y_factor)`. Values represent a percentage of the image to shear over.
            For example, 0.3 shears pixels up to 30% of the way across the image.
            All provided values should be positive.  If `None` is passed, no shear
            occurs on the Y axis.
            Defaults to `None`.
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
        bounding_box_format: The format of bounding boxes of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
    """

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
        seed=None,
        bounding_box_format=None,
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
                "RandomShear received both `x_factor=None` and `y_factor=None`.  As a "
                "result, the layer will perform no augmentation."
            )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation(self, **kwargs):
        x = self._get_shear_amount(self.x_factor)
        y = self._get_shear_amount(self.y_factor)
        return (x, y)

    def _get_shear_amount(self, constraint):
        if constraint is None:
            return None

        invert = preprocessing.random_inversion(self._random_generator)
        return invert * constraint()

    def augment_image(self, image, transformation=None, **kwargs):
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

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomShear()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomShear(bounding_box_format='xyxy')`"
            )
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=image,
            dtype=self.compute_dtype,
        )
        x, y = transformation
        extended_bboxes, rest_axes = self._convert_to_extended_corners_format(
            bounding_boxes
        )
        if x is not None:
            extended_bboxes = self._apply_horizontal_transformation_to_bounding_box(
                extended_bboxes, x
            )
        # apply vertical shear
        if y is not None:
            extended_bboxes = self._apply_vertical_transformation_to_bounding_box(
                extended_bboxes, y
            )

        bounding_boxes = self._convert_to_four_coordinate(extended_bboxes, x, y)
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, images=image, bounding_box_format="xyxy"
        )
        # join rest of the axes with bbox axes
        bounding_boxes = tf.concat(
            [bounding_boxes, rest_axes],
            axis=-1,
        )
        # convert to universal output format
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            images=image,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

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

    @staticmethod
    def _format_transform(transform):
        transform = tf.convert_to_tensor(transform, dtype=tf.float32)
        return transform[tf.newaxis]

    @staticmethod
    def _convert_to_four_coordinate(extended_bboxes, x, y):
        """convert from extended coordinates to 4 coordinates system"""
        (
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
            top_right_x,
            top_right_y,
            bottom_left_x,
            bottom_left_y,
        ) = tf.split(extended_bboxes, 8, axis=1)

        # choose x1,x2 when x>0
        def positive_case_x():
            final_x1 = bottom_left_x
            final_x2 = top_right_x
            return final_x1, final_x2

        # choose x1,x2 when x<0
        def negative_case_x():
            final_x1 = top_left_x
            final_x2 = bottom_right_x
            return final_x1, final_x2

        if x is not None:
            final_x1, final_x2 = tf.cond(
                tf.less(x, 0), negative_case_x, positive_case_x
            )
        else:
            final_x1, final_x2 = top_left_x, bottom_right_x

        # choose y1,y2 when y > 0
        def positive_case_y():
            final_y1 = top_right_y
            final_y2 = bottom_left_y
            return final_y1, final_y2

        # choose y1,y2 when y < 0
        def negative_case_y():
            final_y1 = top_left_y
            final_y2 = bottom_right_y
            return final_y1, final_y2

        if y is not None:
            final_y1, final_y2 = tf.cond(
                tf.less(y, 0), negative_case_y, positive_case_y
            )
        else:
            final_y1, final_y2 = top_left_y, bottom_right_y
        return tf.concat(
            [final_x1, final_y1, final_x2, final_y2],
            axis=1,
        )

    @staticmethod
    def _apply_horizontal_transformation_to_bounding_box(extended_bounding_boxes, x):
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, -x, 0, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(extended_bounding_boxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        return transformed_bboxes

    @staticmethod
    def _apply_vertical_transformation_to_bounding_box(extended_bounding_boxes, y):
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, 0, -y, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(extended_bounding_boxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        return transformed_bboxes

    @staticmethod
    def _convert_to_extended_corners_format(bounding_boxes):
        """splits corner bboxes top left,bottom right to 4 corners top left,
        bottom right,top right and bottom left"""
        x1, y1, x2, y2, rest = tf.split(
            bounding_boxes, [1, 1, 1, 1, bounding_boxes.shape[-1] - 4], axis=-1
        )
        new_bboxes = tf.concat(
            [x1, y1, x2, y2, x2, y1, x1, y2],
            axis=-1,
        )
        return new_bboxes, rest
