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


class RandomRaggedCrop(BaseImageAugmentationLayer):
    """RandomRaggedCrop is an augmentation layer that randomly crops raggedly.

    TODO

    Args:
        height_factor:
        width_factor:
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        bounding_box_format=None,
        interpolation="bilinear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interpolation = keras_cv.utils.get_interpolation(interpolation)
        self.height_factor = keras_cv.utils.parse_factor(
            height_factor, min_value=0.0, max_value=None, param_name="height_factor"
        )
        self.width_factor = keras_cv.utils.parse_factor(
            width_factor, min_value=0.0, max_value=None, param_name="width_factor"
        )
        self.bounding_box_format = bounding_box_format
        self.force_output_ragged_images = True

    def get_random_transformation(self, **kwargs):
        new_height = self.height_factor(dtype=self.compute_dtype)
        new_width = self.width_factor(dtype=self.compute_dtype)

        height_offset = self._random_generator.random_uniform(
            (),
            minval=0.0,
            maxval=tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )
        width_offset = self._random_generator.random_uniform(
            (),
            minval=0.0,
            maxval=tf.maximum(0.0, 1.0 - new_width),
            dtype=tf.float32,
        )

        return {
            "x": width_offset,
            "y": height_offset,
            "width": new_width,
            "height": new_height,
        }

    def compute_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=images.shape[1:],
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
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=image,
        )

        width_offset = transformation["x"]
        height_offset = transformation["y"]
        new_width = transformation["width"]
        new_height = transformation["height"]
        x1, y1, x2, y2, rest = tf.split(
            bounding_boxes, [1, 1, 1, 1, bounding_boxes.shape[-1] - 4], axis=-1
        )

        x1 = (x1 - width_offset) / new_width
        x2 = (x2 - width_offset) / new_width
        y1 = (y1 - height_offset) / new_height
        y2 = (y2 - height_offset) / new_height

        bounding_boxes = tf.concat([x1, y1, x2, y2, rest], axis=-1)
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes, image_shape=(1.0, 1.0, 3), bounding_box_format="rel_xyxy"
        )

        original_shape = tf.cast(tf.shape(image.to_tensor()), self.compute_dtype)

        w = tf.cast(new_width * original_shape[1], tf.int32)
        h = tf.cast(new_height * original_shape[0], tf.int32)
        output_image_shape = (h, w, 3)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            image_shape=output_image_shape,
        )
        return bounding_boxes

    def _crop(self, image, transformation, **kwargs):
        boxes = transformation
        image_shape = tf.cast(tf.shape(image), self.compute_dtype)
        y = tf.cast(image_shape[0] * transformation["y"], tf.int32)
        x = tf.cast(image_shape[1] * transformation["x"], tf.int32)
        height = tf.cast(image_shape[0] * transformation["height"], tf.int32)
        width = tf.cast(image_shape[1] * transformation["width"], tf.int32)
        # tf.print('height', y+height, image_shape[0], y+height > tf.cast(image_shape[0], tf.int32))
        # tf.print('width', x+width, image_shape[1],  x+width > tf.cast(image_shape[1], tf.int32))
        return tf.image.crop_to_bounding_box(image, y, x, height, width)

    def augment_image(self, image, transformation, **kwargs):
        return self._crop(image, transformation)
