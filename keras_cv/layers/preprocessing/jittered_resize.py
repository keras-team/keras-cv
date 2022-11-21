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


class JitteredResize(BaseImageAugmentationLayer):
    """JitteredResize implements MaskRCNN style image augmentation."""

    def __init__(
        self,
        desired_size,
        padded_size,
        scale_factor,
        bounding_box_format=None,
        interpolation="bilinear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.interpolation = keras_cv.utils.get_interpolation(interpolation)
        self.scale_factor = keras_cv.utils.parse_factor(
            scale_factor, min_value=0.0, max_value=None, param_name="scale_factor"
        )
        self.desired_size = desired_size
        self.padded_size = padded_size
        self.bounding_box_format = bounding_box_format
        self.output_dense_images = True

    def get_random_transformation(self, image=None, **kwargs):
        original_image_shape = tf.shape(image)
        image_shape = tf.cast(original_image_shape[0:2], tf.float32)

        scaled_size = tf.round(image_shape * self.scale_factor())
        scale = tf.minimum(
            scaled_size[0] / image_shape[0], scaled_size[1] / image_shape[1]
        )

        scaled_size = tf.round(image_shape * scale)
        image_scale = scaled_size / image_shape

        max_offset = scaled_size - self.desired_size
        max_offset = tf.where(
            tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
        )
        offset = max_offset * tf.random.uniform(
            [
                2,
            ],
            0,
            1,
        )
        offset = tf.cast(offset, tf.int32)

        return {
            "original_size": original_image_shape,
            "image_scale": image_scale,
            "scaled_size": scaled_size,
            "offset": offset,
        }

    def compute_image_signature(self, images):
        return tf.TensorSpec(
            shape=list(self.padded_size) + [images.shape[-1]],
            dtype=self.compute_dtype,
        )

    def augment_image(self, image, transformation, **kwargs):
        # unpackage augmentation arguments
        scaled_size = transformation["scaled_size"]
        offset = transformation["offset"]
        padded_size = self.padded_size
        desired_size = self.desired_size

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=self.interpolation
        )
        scaled_image = scaled_image[
            offset[0] : offset[0] + desired_size[0],
            offset[1] : offset[1] + desired_size[1],
            :,
        ]
        scaled_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )
        return scaled_image

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        if self.bounding_box_format is None:
            raise ValueError(
                "Please provide a `bounding_box_format` when augmenting "
                "bounding boxes with `JitteredResize()`."
            )

        image_scale = tf.cast(transformation["image_scale"], self.compute_dtype)
        offset = tf.cast(transformation["offset"], self.compute_dtype)
        original_size = transformation["original_size"]

        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes,
            image_shape=original_size,
            source=self.bounding_box_format,
            target="yxyx",
        )

        # Adjusts box coordinates based on image_scale and offset.
        yxyx = bounding_boxes[:, :4]
        rest = bounding_boxes[:, 4:]
        yxyx *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        yxyx -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])

        bounding_boxes = tf.concat([yxyx, rest], axis=-1)
        bounding_boxes = keras_cv.bounding_box.clip_to_image(
            bounding_boxes,
            image_shape=self.padded_size + (3,),
            bounding_box_format="yxyx",
        )
        return keras_cv.bounding_box.convert_format(
            bounding_boxes,
            image_shape=self.padded_size + (3,),
            source="yxyx",
            target=self.bounding_box_format,
        )
