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

import keras_cv
from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


class RandomScale(BaseImageAugmentationLayer):
    """RandomScale is an augmentation layer that random scales each image by a factor.

    RandomScale accepts a batch of inputs, and outputs a batch of RaggedTensor images.
    This layer is useful when training models that are meant to be agnostic to input
    image size.

    Args:
        scale_factor: A tuple of two floats or keras_cv.FactorSampler. Represents the
            amount to scale each image by.  The resulting image is output size is
            (height*factor, width*factor) based on the factor sampled from
            `scale_factor`.
        aspect_ratio_factor: TODO(lukewood):
        interpolation: interpolation method used in the `Resize` op.
             Supported values are `"nearest"` and `"bilinear"`.
             Defaults to `"bilinear"`.
    """

    def __init__(
        self, factor, interpolation="bilinear", bounding_box_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.interpolation = keras_cv.utils.get_interpolation(interpolation)
        self.factor = keras_cv.utils.parse_factor(
            factor, min_value=0.0, max_value=None, param_name="factor"
        )
        self.bounding_box_format = bounding_box_format
        self.force_output_ragged_images = True

    def get_random_transformation(self, **kwargs):
        return self.factor(dtype=self.compute_dtype)

    def compute_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=(None, None, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def augment_bounding_boxes(self, bounding_boxes, transformation, image, **kwargs):
        if self.bounding_box_format is None:
            raise ValueError(
                "Please provide a `bounding_box_format` when augmenting "
                "bounding boxes with `RandomScale()`."
            )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes, source=self.bounding_box_format, target="xyxy", images=image
        )
        bounding_boxes = bounding_boxes * transformation
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes, source="xyxy", target=self.bounding_box_format, images=image
        )
        return bounding_boxes

    def augment_image(self, image, transformation, **kwargs):
        target_size = tf.cast(
            tf.cast(tf.shape(image)[:2], self.compute_dtype) * transformation, tf.int32
        )
        return tf.image.resize(image, size=target_size, method=self.interpolation)
