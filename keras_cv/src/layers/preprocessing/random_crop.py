# Copyright 2023 The KerasCV Authors
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
from keras_cv.src import layers as cv_layers
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.RandomCrop")
class RandomCrop(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly crops images.

    This layer will randomly choose a location to crop images down to a target
    size.

    If an input image is smaller than the target size, the input will be
    resized and cropped to return the largest possible window in the image that
    matches the target aspect ratio.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self, height, width, seed=None, bounding_box_format=None, **kwargs
    ):
        super().__init__(
            **kwargs,
            autocast=False,
            seed=seed,
        )
        self.height = height
        self.width = width
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        self.force_output_dense_images = True

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def get_random_transformation_batch(self, batch_size, **kwargs):
        tops = tf.cast(
            self._random_generator.uniform(
                shape=(batch_size, 1), minval=0, maxval=1
            ),
            self.compute_dtype,
        )
        lefts = tf.cast(
            self._random_generator.uniform(
                shape=(batch_size, 1), minval=0, maxval=1
            ),
            self.compute_dtype,
        )
        return {"tops": tops, "lefts": lefts}

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        tops = transformation["tops"]
        lefts = transformation["lefts"]
        transformation = {
            "tops": tf.expand_dims(tops, axis=0),
            "lefts": tf.expand_dims(lefts, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        batch_size = tf.shape(images)[0]
        channel = tf.shape(images)[-1]
        heights, widths = self._get_image_shape(images)
        h_diffs = heights - self.height
        w_diffs = widths - self.width
        # broadcast
        h_diffs = (
            tf.ones(
                shape=(batch_size, self.height, self.width, channel),
                dtype=tf.int32,
            )
            * h_diffs[:, tf.newaxis, tf.newaxis, :]
        )
        w_diffs = (
            tf.ones(
                shape=(batch_size, self.height, self.width, channel),
                dtype=tf.int32,
            )
            * w_diffs[:, tf.newaxis, tf.newaxis, :]
        )
        return tf.where(
            tf.math.logical_and(h_diffs >= 0, w_diffs >= 0),
            self._crop_images(images, transformations),
            self._resize_images(images),
        )

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCrop(bounding_box_format='xyxy')`"
            )
        if isinstance(bounding_boxes["boxes"], tf.RaggedTensor):
            bounding_boxes = bounding_box.to_dense(
                bounding_boxes, default_value=-1
            )
        batch_size = tf.shape(raw_images)[0]
        heights, widths = self._get_image_shape(raw_images)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )
        h_diffs = heights - self.height
        w_diffs = widths - self.width
        # broadcast
        num_bounding_boxes = tf.shape(bounding_boxes["boxes"])[-2]
        h_diffs = (
            tf.ones(
                shape=(batch_size, num_bounding_boxes, 4),
                dtype=tf.int32,
            )
            * h_diffs[:, tf.newaxis, :]
        )
        w_diffs = (
            tf.ones(
                shape=(batch_size, num_bounding_boxes, 4),
                dtype=tf.int32,
            )
            * w_diffs[:, tf.newaxis, :]
        )
        boxes = tf.where(
            tf.math.logical_and(h_diffs >= 0, w_diffs >= 0),
            self._crop_bounding_boxes(
                raw_images, bounding_boxes["boxes"], transformations
            ),
            self._resize_bounding_boxes(
                raw_images,
                bounding_boxes["boxes"],
            ),
        )
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            image_shape=(self.height, self.width, None),
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            image_shape=(self.height, self.width, None),
        )
        return bounding_boxes

    def _get_image_shape(self, images):
        if isinstance(images, tf.RaggedTensor):
            heights = tf.reshape(images.row_lengths(), (-1, 1))
            widths = tf.reshape(
                tf.reduce_max(images.row_lengths(axis=2), 1), (-1, 1)
            )
        else:
            batch_size = tf.shape(images)[0]
            heights = tf.repeat(tf.shape(images)[H_AXIS], repeats=[batch_size])
            heights = tf.reshape(heights, shape=(-1, 1))
            widths = tf.repeat(tf.shape(images)[W_AXIS], repeats=[batch_size])
            widths = tf.reshape(widths, shape=(-1, 1))
        return tf.cast(heights, dtype=tf.int32), tf.cast(widths, dtype=tf.int32)

    def _crop_images(self, images, transformations):
        batch_size = tf.shape(images)[0]
        heights, widths = self._get_image_shape(images)
        heights = tf.cast(heights, dtype=self.compute_dtype)
        widths = tf.cast(widths, dtype=self.compute_dtype)

        tops = transformations["tops"]
        lefts = transformations["lefts"]
        x1s = lefts * (widths - self.width)
        y1s = tops * (heights - self.height)

        x2s = x1s + self.width
        y2s = y1s + self.height
        # normalize
        x1s /= widths
        y1s /= heights
        x2s /= widths
        y2s /= heights
        boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)

        images = tf.image.crop_and_resize(
            tf.cast(images, tf.float32),
            tf.cast(boxes, tf.float32),
            tf.range(batch_size),
            [self.height, self.width],
            method="nearest",
        )
        return tf.cast(images, dtype=self.compute_dtype)

    def _resize_images(self, images):
        resizing_layer = cv_layers.Resizing(self.height, self.width)
        outputs = resizing_layer(images)
        return tf.cast(outputs, dtype=self.compute_dtype)

    def _crop_bounding_boxes(self, images, boxes, transformation):
        tops = transformation["tops"]
        lefts = transformation["lefts"]
        heights, widths = self._get_image_shape(images)
        heights = tf.cast(heights, dtype=self.compute_dtype)
        widths = tf.cast(widths, dtype=self.compute_dtype)

        # compute offsets for xyxy bounding_boxes
        top_offsets = tf.cast(
            tf.math.round(tops * (heights - self.height)),
            dtype=self.compute_dtype,
        )
        left_offsets = tf.cast(
            tf.math.round(lefts * (widths - self.width)),
            dtype=self.compute_dtype,
        )

        x1s, y1s, x2s, y2s = tf.split(
            tf.cast(boxes, self.compute_dtype), 4, axis=-1
        )
        x1s -= tf.expand_dims(left_offsets, axis=1)
        y1s -= tf.expand_dims(top_offsets, axis=1)
        x2s -= tf.expand_dims(left_offsets, axis=1)
        y2s -= tf.expand_dims(top_offsets, axis=1)
        outputs = tf.concat([x1s, y1s, x2s, y2s], axis=-1)
        return outputs

    def _resize_bounding_boxes(self, images, boxes):
        heights, widths = self._get_image_shape(images)
        heights = tf.cast(heights, dtype=self.compute_dtype)
        widths = tf.cast(widths, dtype=self.compute_dtype)
        x_scale = tf.cast(self.width / widths, dtype=self.compute_dtype)
        y_scale = tf.cast(self.height / heights, dtype=self.compute_dtype)
        x1s, y1s, x2s, y2s = tf.split(
            tf.cast(boxes, self.compute_dtype), 4, axis=-1
        )
        outputs = tf.concat(
            [
                x1s * x_scale[:, tf.newaxis, :],
                y1s * y_scale[:, tf.newaxis, :],
                x2s * x_scale[:, tf.newaxis, :],
                y2s * y_scale[:, tf.newaxis, :],
            ],
            axis=-1,
        )
        return outputs

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
