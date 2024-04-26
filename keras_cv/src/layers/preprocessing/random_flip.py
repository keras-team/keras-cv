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
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2

# Defining modes for random flipping
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@keras_cv_export("keras_cv.layers.RandomFlip")
class RandomFlip(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly flips images.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`, defaults to
            `"horizontal"`. `"horizontal"` is a left-right flip and
            `"vertical"` is a top-bottom flip.
        rate: A float that controls the frequency of flipping. 1.0 indicates
            that images are always flipped. 0.0 indicates no flipping.
            Defaults to 0.5.
        seed: Integer. Used to create a random seed.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer to
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
    """  # noqa: E501

    def __init__(
        self,
        mode=HORIZONTAL,
        rate=0.5,
        seed=None,
        bounding_box_format=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.mode = mode
        self.seed = seed
        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError(
                "RandomFlip layer {name} received an unknown mode="
                "{arg}".format(name=self.name, arg=mode)
            )
        self.bounding_box_format = bounding_box_format
        if rate < 0.0 or rate > 1.0:
            raise ValueError(
                f"`rate` should be inside of range [0, 1]. Got rate={rate}"
            )
        self.rate = rate

    def get_random_transformation_batch(self, batch_size, **kwargs):
        flip_horizontals = tf.zeros(shape=(batch_size, 1))
        flip_verticals = tf.zeros(shape=(batch_size, 1))

        if self.horizontal:
            flip_horizontals = self._random_generator.uniform(
                shape=(batch_size, 1)
            )

        if self.vertical:
            flip_verticals = self._random_generator.uniform(
                shape=(batch_size, 1)
            )

        return {
            "flip_horizontals": flip_horizontals,
            "flip_verticals": flip_verticals,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        flip_horizontals = transformation["flip_horizontals"]
        flip_verticals = transformation["flip_verticals"]
        transformation = {
            "flip_horizontals": tf.expand_dims(flip_horizontals, axis=0),
            "flip_verticals": tf.expand_dims(flip_verticals, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        return self._flip_images(images, transformations)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations=None, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomFlip()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomFlip(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=raw_images,
        )
        boxes = bounding_boxes["boxes"]
        batch_size = tf.shape(boxes)[0]
        max_boxes = tf.shape(boxes)[1]
        flip_horizontals = transformations["flip_horizontals"]
        flip_verticals = transformations["flip_verticals"]

        # broadcast
        flip_horizontals = (
            tf.ones(shape=(batch_size, max_boxes, 4))
            * flip_horizontals[:, tf.newaxis, :]
        )
        flip_verticals = (
            tf.ones(shape=(batch_size, max_boxes, 4))
            * flip_verticals[:, tf.newaxis, :]
        )

        boxes = tf.where(
            flip_horizontals > (1.0 - self.rate),
            self._flip_boxes_horizontal(boxes),
            boxes,
        )
        boxes = tf.where(
            flip_verticals > (1.0 - self.rate),
            self._flip_boxes_vertical(boxes),
            boxes,
        )

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=raw_images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations=None, **kwargs
    ):
        return self._flip_images(segmentation_masks, transformations)

    def _flip_images(self, images, transformations):
        batch_size = tf.shape(images)[0]
        height, width = tf.shape(images)[1], tf.shape(images)[2]
        channel = tf.shape(images)[3]
        flip_horizontals = transformations["flip_horizontals"]
        flip_verticals = transformations["flip_verticals"]

        # broadcast
        flip_horizontals = (
            tf.ones(shape=(batch_size, height, width, channel))
            * flip_horizontals[:, tf.newaxis, tf.newaxis, :]
        )
        flip_verticals = (
            tf.ones(shape=(batch_size, height, width, channel))
            * flip_verticals[:, tf.newaxis, tf.newaxis, :]
        )

        flipped_outputs = tf.where(
            flip_horizontals > (1.0 - self.rate),
            tf.image.flip_left_right(images),
            images,
        )
        flipped_outputs = tf.where(
            flip_verticals > (1.0 - self.rate),
            tf.image.flip_up_down(flipped_outputs),
            flipped_outputs,
        )
        flipped_outputs.set_shape(images.shape)
        return flipped_outputs

    def _flip_boxes_horizontal(self, boxes):
        x1, x2, x3, x4 = tf.split(boxes, 4, axis=-1)
        outputs = tf.concat([1 - x3, x2, 1 - x1, x4], axis=-1)
        return outputs

    def _flip_boxes_vertical(self, boxes):
        x1, x2, x3, x4 = tf.split(boxes, 4, axis=-1)
        outputs = tf.concat([x1, 1 - x4, x3, 1 - x2], axis=-1)
        return outputs

    def get_config(self):
        config = {
            "mode": self.mode,
            "rate": self.rate,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
