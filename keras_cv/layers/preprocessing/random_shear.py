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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
import keras_cv
from keras_cv.utils import preprocessing



@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomShear(BaseImageAugmentationLayer):
    """Randomly shears an image.

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
        bounding_box_format : Specify input bounding box format.
        Supported formats : xyxy, rel_xyxy, xywh, center_xywh
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
                "seed": self.seed,
            }
        )
        return config

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        image = None
        image = kwargs.get("image")
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomShear()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomShear(bounding_box_format='xyxy')`"
            )
        height, width, _ = image.shape
        image = tf.expand_dims(image, axis=0)

        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=image,
            dtype=self.compute_dtype,
        )
        x, y = transformation
        # apply horizontal shear
        if x is not None:
            bounding_boxes = self._apply_horizontal_transformation_to_bounding_box(
                bounding_boxes, x
            )
        # apply vertical shear
        if y is not None:
            bounding_boxes = self._apply_vertical_transformation_to_bounding_box(
                bounding_boxes, y
            )
        # clip bounding boxes value to 0-image height and 0-image width
        bounding_boxes = RandomShear.clip_bounding_box(height, width, bounding_boxes)
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            images=image,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    @staticmethod
    def clip_bounding_box(height, width, bounding_boxes):
        """clips bounding boxes b/w 0 - image width and 0 - image height"""
        x1, y1, x2, y2 = tf.split(bounding_boxes, 4, axis=1)
        new_bboxes = tf.stack(
            [
                tf.clip_by_value(x1, clip_value_min=0, clip_value_max=width),
                tf.clip_by_value(y1, clip_value_min=0, clip_value_max=height),
                tf.clip_by_value(x2, clip_value_min=0, clip_value_max=width),
                tf.clip_by_value(y2, clip_value_min=0, clip_value_max=height),
            ],
            axis=1,
        )
        new_bboxes = tf.squeeze(new_bboxes, axis=-1)
        return new_bboxes

    @staticmethod
    def convert_to_extended_corners_format(bounding_boxes):
        """splits corner bboxes top left,bottom right to 4 corners top left,
        bottom right,top right and bottom left"""
        x1, y1, x2, y2 = tf.split(bounding_boxes, 4, axis=1)
        new_bboxes = tf.stack(
            [
                x1,
                y1,
                x2,
                y2,
                x2,
                y1,
                x1,
                y2,
            ],
            axis=1,
        )
        return new_bboxes

    def _apply_horizontal_transformation_to_bounding_box(self, bounding_boxes, x):
        """args: image : takes a single image H,W,C,
        bounding_boxes: take bbox coordinates [N,4] -> [x1,y1,x2,y2]
        x: x transformation None if no transformation"""
        new_bboxes = self.convert_to_extended_corners_format(bounding_boxes)
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, -x, 0, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(new_bboxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        # split into 4 corners of bbox
        (
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
            top_right_x,
            top_right_y,
            bottom_left_x,
            bottom_left_y,
        ) = tf.split(transformed_bboxes, 8, axis=1)

        # choose x1,x2 when x>0
        def positive_case():
            final_x1 = bottom_left_x
            final_x2 = top_right_x
            return final_x1, final_x2

        # choose x1,x2 when x<0
        def negative_case():
            final_x1 = top_left_x
            final_x2 = bottom_right_x
            return final_x1, final_x2

        final_x1, final_x2 = tf.cond(tf.less(x, 0), negative_case, positive_case)
        return tf.concat(
            [final_x1, top_left_y, final_x2, bottom_right_y],
            axis=1,
        )

    def _apply_vertical_transformation_to_bounding_box(self, bounding_boxes, y):
        """args: image : takes a single image H,W,C,
        bounding_boxes: take bbox coordinates [N,4] -> [y1,x1,y2,x2]
        y: y transformation None if no transformation"""
        new_bboxes = self.convert_to_extended_corners_format(bounding_boxes)
        # create transformation matrix [1,4]
        matrix = tf.stack([1.0, 0, -y, 1.0], axis=0)
        # reshape it to [2,2]
        matrix = tf.reshape(matrix, (2, 2))
        # reshape unnormalized bboxes from [N,8] -> [N*4,2]
        new_bboxes = tf.reshape(new_bboxes, (-1, 2))
        # [[1,x`],[y`,1]]*[x,y]->[new_x,new_y]
        transformed_bboxes = tf.reshape(
            tf.einsum("ij,kj->ki", matrix, new_bboxes), (-1, 8)
        )
        # split into 4 corners of bbox
        (
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
            top_right_x,
            top_right_y,
            bottom_left_x,
            bottom_left_y,
        ) = tf.split(transformed_bboxes, 8, axis=1)

        # choose y1,y2 when y > 0
        def positive_case():
            final_y1 = top_right_y
            final_y2 = bottom_left_y
            return final_y1, final_y2

        # choose y1,y2 when y < 0
        def negative_case():
            final_y1 = top_left_y
            final_y2 = bottom_right_y
            return final_y1, final_y2

        final_y1, final_y2 = tf.cond(tf.less(y, 0), negative_case, positive_case)
        return tf.concat(
            [top_left_x, final_y1, top_right_x, final_y2],
            axis=1,
        )
