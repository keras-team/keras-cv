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

from keras_cv import bounding_box
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2

# Defining modes for random flipping
HORIZONTAL = "horizontal"
VERTICAL = "vertical"
HORIZONTAL_AND_VERTICAL = "horizontal_and_vertical"


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomFlip(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Arguments:
      mode: String indicating which flip mode to use. Can be `"horizontal"`,
        `"vertical"`, or `"horizontal_and_vertical"`. Defaults to
        `"horizontal_and_vertical"`. `"horizontal"` is a left-right flip and
        `"vertical"` is a top-bottom flip.
      seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        mode=HORIZONTAL_AND_VERTICAL,
        seed=None,
        bounding_box_format=None,
        **kwargs
    ):
        super().__init__(seed=seed, force_generator=True, **kwargs)
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
        self.auto_vectorize = True
        self.bounding_box_format = bounding_box_format

    def augment_label(self, label, transformation, **kwargs):
        return label

    def augment_image(self, image, transformation, **kwargs):
        flipped_output = tf.cond(
            transformation["flip_horizontal"],
            lambda: tf.image.flip_left_right(image),
            lambda: image,
        )
        flipped_output = tf.cond(
            transformation["flip_vertical"],
            lambda: tf.image.flip_up_down(flipped_output),
            lambda: flipped_output,
        )
        flipped_output.set_shape(image.shape)
        return flipped_output

    def get_random_transformation(self, **kwargs):
        flip_horizontal = False
        flip_vertical = False
        if self.horizontal:
            flip_horizontal = self._random_generator.random_uniform(shape=[]) > 0.5
        if self.vertical:
            flip_vertical = self._random_generator.random_uniform(shape=[]) > 0.5
        return {
            "flip_horizontal": tf.cast(flip_horizontal, dtype=tf.bool),
            "flip_vertical": tf.cast(flip_vertical, dtype=tf.bool),
        }

    def _flip_bounding_boxes_horizontal(bounding_boxes):
        return tf.stack(
            [
                1 - bounding_boxes[:, 2],
                bounding_boxes[:, 1],
                1 - bounding_boxes[:, 0],
                bounding_boxes[:, 3],
            ],
            axis=-1,
        )

    def _flip_bounding_boxes_vertical(bounding_boxes):
        return tf.stack(
            [
                bounding_boxes[:, 0],
                1 - bounding_boxes[:, 3],
                bounding_boxes[:, 2],
                1 - bounding_boxes[:, 1],
            ],
            axis=-1,
        )

    def augment_bounding_boxes(
        self, bounding_boxes, transformation=None, image=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomFlip()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomFlip(bounding_box_format='xyxy')`"
            )

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=image,
        )
        bounding_boxes = tf.cond(
            transformation["flip_horizontal"],
            lambda: RandomFlip._flip_bounding_boxes_horizontal(bounding_boxes),
            lambda: bounding_boxes,
        )
        bounding_boxes = tf.cond(
            transformation["flip_vertical"],
            lambda: RandomFlip._flip_bounding_boxes_vertical(bounding_boxes),
            lambda: bounding_boxes,
        )
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=image,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "mode": self.mode,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
