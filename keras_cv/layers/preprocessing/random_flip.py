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
from keras_cv import utils
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

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

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
        self.auto_vectorize = False
        self.bounding_box_format = bounding_box_format

    def augment_label(self, label, transformation, **kwargs):
        return label

    def augment_image(self, image, transformation, **kwargs):
        image = utils.preprocessing.ensure_tensor(image, self.compute_dtype)
        flipped_output = image
        if transformation["flip_horizontal"]:
            flipped_output = tf.image.flip_left_right(flipped_output)
        if transformation["flip_vertical"]:
            flipped_output = tf.image.flip_up_down(flipped_output)
        flipped_output.set_shape(image.shape)
        return flipped_output

    def get_random_transformation(self, **kwargs):
        flip_horizontal = False
        flip_vertical = False
        if self.horizontal:
            flip_horizontal = (
                True
                if (tf.random.uniform(shape=[], minval=0, maxval=1) > 0.5)
                else False
            )
        if self.vertical:
            a = tf.random.uniform(shape=[], minval=0, maxval=1)
            flip_vertical = True if a > 0.5 else False
        return {
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
        }

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
            target="xyxy",
            images=image,
        )
        image_shape = tf.shape(image)
        h = tf.cast(image_shape[H_AXIS], dtype="float32")
        w = tf.cast(image_shape[W_AXIS], dtype="float32")
        bounding_boxes_out = tf.identity(bounding_boxes)
        if transformation["flip_horizontal"]:
            bounding_boxes_out = tf.stack(
                [
                    w - bounding_boxes_out[:, 2],
                    bounding_boxes_out[:, 1],
                    w - bounding_boxes_out[:, 0],
                    bounding_boxes_out[:, 3],
                ],
                axis=-1,
            )
        if transformation["flip_vertical"]:
            bounding_boxes_out = tf.stack(
                [
                    bounding_boxes_out[:, 0],
                    h - bounding_boxes_out[:, 3],
                    bounding_boxes_out[:, 2],
                    h - bounding_boxes_out[:, 1],
                ],
                axis=-1,
            )
        bounding_boxes_out = bounding_box.clip_to_image(
            bounding_boxes_out,
            bounding_box_format="xyxy",
            images=image,
        )
        bounding_boxes_out = bounding_box.convert_format(
            bounding_boxes_out,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
        )
        return bounding_boxes_out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "mode": self.mode,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
