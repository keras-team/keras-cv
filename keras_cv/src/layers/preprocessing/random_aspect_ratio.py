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
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.utils import get_interpolation
from keras_cv.src.utils import parse_factor


@keras_cv_export("keras_cv.layers.RandomAspectRatio")
class RandomAspectRatio(BaseImageAugmentationLayer):
    """RandomAspectRatio randomly distorts the aspect ratio of the provided
    image.

    This is done on an element-wise basis, and as a consequence this layer
    always returns a tf.RaggedTensor.

    Args:
        factor: a range of values in the range `(0, infinity)` that determines
            the percentage to distort the aspect ratio of each image by.
        interpolation: interpolation method used in the `Resize` op.
             Supported values are `"nearest"` and `"bilinear"`.
             Defaults to `"bilinear"`.
    """

    def __init__(
        self,
        factor,
        interpolation="bilinear",
        bounding_box_format=None,
        seed=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interpolation = get_interpolation(interpolation)
        self.factor = parse_factor(
            factor,
            min_value=0.0,
            max_value=None,
            seed=seed,
            param_name="factor",
        )
        self.bounding_box_format = bounding_box_format
        self.seed = seed
        self.auto_vectorize = False
        self.force_output_ragged_images = True

    def get_random_transformation(self, **kwargs):
        return self.factor(dtype=self.compute_dtype)

    def compute_image_signature(self, images):
        return tf.RaggedTensorSpec(
            shape=(None, None, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )

    def augment_bounding_boxes(
        self, bounding_boxes, transformation, image, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "Please provide a `bounding_box_format` when augmenting "
                "bounding boxes with `RandomAspectRatio()`."
            )
        bounding_boxes = bounding_boxes.copy()
        img_shape = tf.shape(image)
        img_shape = tf.cast(img_shape, self.compute_dtype)
        height, width = img_shape[0], img_shape[1]
        height = height / transformation
        width = width * transformation

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            image_shape=img_shape,
        )
        x, y, x2, y2 = tf.split(bounding_boxes["boxes"], [1, 1, 1, 1], axis=-1)
        x = x * transformation
        x2 = x2 * transformation
        y = y / transformation
        y2 = y2 / transformation
        boxes = tf.concat([x, y, x2, y2], axis=-1)
        boxes = bounding_box.convert_format(
            boxes,
            source="xyxy",
            target=self.bounding_box_format,
            image_shape=tf.stack([height, width, 3], axis=0),
        )
        bounding_boxes["boxes"] = boxes
        return bounding_boxes

    def augment_image(self, image, transformation, **kwargs):
        # images....transformation
        img_shape = tf.cast(tf.shape(image), self.compute_dtype)
        height, width = img_shape[0], img_shape[1]
        height = height / transformation
        width = width * transformation

        target_size = tf.cast(tf.stack([height, width]), tf.int32)
        result = tf.image.resize(
            image, size=target_size, method=self.interpolation
        )
        return tf.cast(result, self.compute_dtype)

    def augment_label(self, label, transformation, **kwargs):
        return label

    def get_config(self):
        config = {
            "factor": self.factor,
            "interpolation": self.interpolation,
            "bounding_box_format": self.bounding_box_format,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
