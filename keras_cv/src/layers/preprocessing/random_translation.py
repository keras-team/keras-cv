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
from keras_cv.src.utils import preprocessing as preprocessing_utils

H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.RandomTranslation")
class RandomTranslation(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly translates images.

    This layer will apply random translations to each image, filling empty
    space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Args:
      height_factor: a float represented as fraction of value, or a tuple of
          size 2 representing lower and upper bound for shifting vertically. A
          negative value means shifting image up, while a positive value means
          shifting image down. When represented as a single positive float, this
          value is used for both the upper and lower bound. For instance,
          `height_factor=(-0.2, 0.3)` results in an output shifted by a random
          amount in the range `[-20%, +30%]`. `height_factor=0.2` results in an
          output height shifted by a random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
          2 representing lower and upper bound for shifting horizontally. A
          negative value means shifting image left, while a positive value means
          shifting image right. When represented as a single positive float,
          this value is used for both the upper and lower bound. For instance,
          `width_factor=(-0.2, 0.3)` results in an output shifted left by 20%,
          and shifted right by 30%. `width_factor=0.2` results
          in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
          to the given mode
          (one of `{"constant", "reflect", "wrap", "nearest"}`).
          - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
              reflecting about the edge of the last pixel.
          - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
              filling all values beyond the edge with the same constant value
              k = 0.
          - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
              wrapping around to the opposite edge.
          - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
              the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
          `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
          boundaries when `fill_mode="constant"`.
      bounding_box_format: The format of bounding boxes of input dataset.
          Refer to
          https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
          for more details on supported bounding box formats. This is required
          when augmenting data which includes bounding boxes.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`,  in `"channels_last"` format.
    """

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError(
                "`height_factor` cannot have upper bound less than "
                f"lower bound, got {height_factor}"
            )
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` must have values between [-1, 1], "
                f"got {height_factor}"
            )

        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError(
                "`width_factor` cannot have upper bound less than "
                f"lower bound, got {width_factor}"
            )
        if abs(self.width_lower) > 1.0 or abs(self.width_upper) > 1.0:
            raise ValueError(
                "`width_factor` must have values between [-1, 1], "
                f"got {width_factor}"
            )

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation_batch(self, batch_size, **kwargs):
        height_translations = self._random_generator.uniform(
            shape=[batch_size, 1],
            minval=self.height_lower,
            maxval=self.height_upper,
            dtype=tf.float32,
        )
        width_translations = self._random_generator.uniform(
            shape=[batch_size, 1],
            minval=self.width_lower,
            maxval=self.width_upper,
            dtype=tf.float32,
        )
        return {
            "height_translations": height_translations,
            "width_translations": width_translations,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        height_translations = transformation["height_translations"]
        width_translations = transformation["width_translations"]
        transformation = {
            "height_translations": tf.expand_dims(height_translations, axis=0),
            "width_translations": tf.expand_dims(width_translations, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        """Translated inputs with random ops."""
        original_shape = images.shape
        inputs_shape = tf.shape(images)
        img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
        height_translations = transformations["height_translations"]
        width_translations = transformations["width_translations"]
        height_translations = height_translations * img_hd
        width_translations = width_translations * img_wd
        translations = tf.cast(
            tf.concat([width_translations, height_translations], axis=1),
            dtype=tf.float32,
        )
        output = preprocessing_utils.transform(
            images,
            preprocessing_utils.get_translation_matrix(translations),
            interpolation=self.interpolation,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        output.set_shape(original_shape)
        return output

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        segmentation_masks = preprocessing_utils.ensure_tensor(
            segmentation_masks, self.compute_dtype
        )
        original_shape = segmentation_masks.shape
        mask_shape = tf.shape(segmentation_masks)
        img_hd = tf.cast(mask_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(mask_shape[W_AXIS], tf.float32)
        height_translations = transformations["height_translations"]
        width_translations = transformations["width_translations"]
        height_translations = height_translations * img_hd
        width_translations = width_translations * img_wd
        translations = tf.cast(
            tf.concat([width_translations, height_translations], axis=1),
            dtype=tf.float32,
        )
        output = preprocessing_utils.transform(
            segmentation_masks,
            preprocessing_utils.get_translation_matrix(translations),
            interpolation="nearest",
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
        )
        output.set_shape(original_shape)
        return output

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomTranslation()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomTranslation(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=images,
            dtype=self.compute_dtype,
        )

        boxes = bounding_boxes["boxes"]
        x1, y1, x2, y2 = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x1 += tf.expand_dims(transformations["width_translations"], axis=1)
        x2 += tf.expand_dims(transformations["width_translations"], axis=1)
        y1 += tf.expand_dims(transformations["height_translations"], axis=1)
        y2 += tf.expand_dims(transformations["height_translations"], axis=1)

        bounding_boxes["boxes"] = tf.concat([x1, y1, x2, y2], axis=-1)
        bounding_boxes = bounding_box.to_dense(bounding_boxes)

        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=images,
        )
        bounding_boxes = bounding_box.to_ragged(bounding_boxes)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            images=images,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
