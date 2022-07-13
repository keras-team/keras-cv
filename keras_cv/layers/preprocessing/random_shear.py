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
import functools
import warnings

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import keypoint
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.layers.preprocessing.random_rotation import MissingFormatError
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomShear(BaseImageAugmentationLayer):
    """Randomly shears an image.

    Args:
      x_factor: A tuple of two floats, a single float or a
       `keras_cv.FactorSampler`. For each augmented image a value is
       sampled from the provided range. If a float is passed, the
       range is interpreted as `(0, x_factor)`.  Values represent a
       percentage of the image to shear over.  For example, 0.3 shears
       pixels up to 30% of the way across the image.  All provided
       values should be positive.  If `None` is passed, no shear
       occurs on the X axis.  Defaults to `None`.
      y_factor: A tuple of two floats, a single float or a
       `keras_cv.FactorSampler`. For each augmented image a value is
       sampled from the provided range. If a float is passed, the
       range is interpreted as `(0, y_factor)`. Values represent a
       percentage of the image to shear over.  For example, 0.3 shears
       pixels up to 30% of the way across the image.  All provided
       values should be positive.  If `None` is passed, no shear
       occurs on the Y axis.  Defaults to `None`.
      interpolation: interpolation method used in the
        `ImageProjectiveTransformV3` op.  Supported values are
        `"nearest"` and `"bilinear"`.  Defaults to `"bilinear"`.
      fill_mode: fill_mode in the `ImageProjectiveTransformV3` op.
        Supported values are `"reflect"`, `"wrap"`, `"constant"`, and
        `"nearest"`.  Defaults to `"reflect"`.
      fill_value: fill_value in the `ImageProjectiveTransformV3` op.
        A `Tensor` of type `float32`. The value to be filled when
        fill_mode is constant".  Defaults to `0.0`.
      bounding_box_format: The format of bounding boxes of input
        dataset. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      keypoint_format: The format of keypoints of input dataset. Refer
        to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported keypoint formats.
      clip_points_to_image_size: Indicates if bounding boxes and
        keypoints should respectively be clipped or discarded
        according to the input image shape. Note that you should
        disable clipping while augmenting batches of images with
        keypoints data.
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
        bounding_box_format=None,
        keypoint_format=None,
        clip_points_to_image_size=True,
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
        self.keypoint_format = keypoint_format
        self.clip_points_to_image_size = clip_points_to_image_size

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

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        image = kwargs.get("image")
        if self.bounding_box_format is None:
            MissingFormatError.bounding_boxes("RandomShear")

        return bounding_box.transform_from_point_transform(
            bounding_boxes,
            functools.partial(
                self.augment_keypoints,
                transformation=transformation,
                image=image,
                keypoint_format="xy",
                discard_out_of_image=False,
            ),
            bounding_box_format=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
            clip_boxes=self.clip_points_to_image_size,
        )

    def augment_keypoints(
        self,
        keypoints,
        image,
        transformation=None,
        **kwargs,
    ):
        discard_out_of_image = kwargs.get(
            "discard_out_of_image", self.clip_points_to_image_size
        )
        keypoint_format = kwargs.get("keypoint_format", self.keypoint_format)
        if keypoint_format is None:
            raise MissingFormatError.keypoints("RandomShear")
        keypoints = keypoint.convert_format(
            keypoints, source=keypoint_format, target="xy", images=image
        )
        x, y = transformation
        if x is not None:
            offset_x = keypoints[..., 1] * x
            offset_y = tf.zeros_like(offset_x)
            keypoints = keypoints - tf.stack([offset_x, offset_y], axis=-1)
        if y is not None:
            offset_y = keypoints[..., 0] * y
            offset_x = tf.zeros_like(offset_y)
            keypoints = keypoints - tf.stack([offset_x, offset_y], axis=-1)

        if discard_out_of_image:
            keypoints = keypoint.discard_out_of_image(keypoints, image)

        return keypoint.convert_format(
            keypoints, source="xy", target=keypoint_format, images=image
        )

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
                "bounding_box_format": self.bounding_box_format,
                "keypoint_format": self.keypoint_format,
                "clip_points_to_image_size": self.clip_points_to_image_size,
            }
        )
        return config
