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
    """

    def __init__(
        self,
        x_factor=None,
        y_factor=None,
        interpolation="bilinear",
        fill_mode="reflect",
        fill_value=0.0,
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
                "RandomShear received both `x_factor=None` and `y_factor=None`.  As a "
                "result, the layer will perform no augmentation."
            )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

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

    def _augment(self, inputs):
        image = inputs.get("images", None)
        label = inputs.get("labels", None)
        bounding_box = inputs.get("bounding_boxes", None)
        transformation = self.get_random_transformation(
            image=image, label=label, bounding_box=bounding_box
        )  # pylint: disable=assignment-from-none
        image = self.augment_image(image, transformation=transformation)
        result = {"images": image}
        if label is not None:
            label = self.augment_label(label, transformation=transformation)
            result["labels"] = label
        if bounding_box is not None:
            # changed augment_bounding_box requires image as well
            # for unnormalizing bbox coordinates
            bounding_box = self.augment_bounding_box(
                image, bounding_box, transformation=transformation
            )
            result["bounding_boxes"] = bounding_box
        return result

    def augment_bounding_box(self, image, bounding_boxes, transformation):
        """args: image : takes a single image H,W,C,
        bounding_boxes: take bbox coordinates [N,4] -> [y1,x1,y2,x2]
        transformation: takes tuple for x,y transformation None if no transformation"""
        x, y = transformation
        if x is not None:
            bounding_boxes = self.augment_horizontal(image, bounding_boxes, x)
        if y is not None:
            bounding_boxes = self.augment_vertical(image, bounding_boxes, y)
        return bounding_boxes

    def augment_horizontal(self, image, bounding_boxes, x):
        """args: image : takes a single image H,W,C,
        bounding_boxes: take bbox coordinates [N,4] -> [y1,x1,y2,x2]
        x: x transformation None if no transformation"""
        height, width, _ = image.shape
        # split bbox coordinates [N,4] to y1,x1,y2,x2 each of shape [N,1]
        y1, x1, y2, x2 = tf.split(bounding_boxes, 4, axis=1)
        # unnormalize bbox coordinates to image shape
        # and tranform into [x1,y1,x2,y2,x3,y3,x4,y4]
        new_bboxes = tf.stack(
            [
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height,
                x2 * width,
                y1 * height,
                x1 * width,
                y2 * height,
            ],
            axis=1,
        )
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
        return tf.clip_by_value(
            tf.concat(
                [
                    top_left_y / height,
                    final_x1 / width,
                    bottom_right_y / height,
                    final_x2 / width,
                ],
                axis=1,
            ),
            clip_value_min=0.0,
            clip_value_max=1.0,
        )

    def augment_vertical(self, image, bounding_boxes, y):
        """args: image : takes a single image H,W,C,
        bounding_boxes: take bbox coordinates [N,4] -> [y1,x1,y2,x2]
        y: y transformation None if no transformation"""
        height, width, _ = image.shape
        # split bbox coordinates [N,4] to y1,x1,y2,x2 each of shape [N,1]
        y1, x1, y2, x2 = tf.split(bounding_boxes, 4, axis=1)
        # unnormalize bbox coordinates to image shape
        # and tranform into [x1,y1,x2,y2,x3,y3,x4,y4]
        new_bboxes = tf.stack(
            [
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height,
                x2 * width,
                y1 * height,
                x1 * width,
                y2 * height,
            ],
            axis=1,
        )
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
        return tf.clip_by_value(
            tf.concat(
                [
                    final_y1 / height,
                    top_left_x / width,
                    final_y2 / height,
                    top_right_x / width,
                ],
                axis=1,
            ),
            clip_value_min=0.0,
            clip_value_max=1.0,
        )
