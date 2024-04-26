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

import numpy as np
import tensorflow as tf

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.RandomRotation")
class RandomRotation(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly rotates images.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format

    Arguments:
      factor: a float represented as fraction of 2 Pi, or a tuple of size 2
        representing lower and upper bound for rotating clockwise and
        counter-clockwise. A positive values means rotating counter clock-wise,
        while a negative value means clock-wise. When represented as a single
        float, this value is used for both the upper and lower bound. For
        instance, `factor=(-0.2, 0.3)` results in an output rotation by a random
        amount in the range `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in
        an output rotating by a random amount in the range
        `[-20% * 2pi, 20% * 2pi]`.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended by
          reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
          filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
          wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by
          the nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the
        boundaries when `fill_mode="constant"`.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      segmentation_classes: an optional integer with the number of classes in
        the input segmentation mask. Required iff augmenting data with sparse
        (non one-hot) segmentation masks. Include the background class in this
        count (e.g. for segmenting dog vs background, this should be set to 2).
    """

    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        segmentation_classes=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError(
                "Factor cannot have negative values, " "got {}".format(factor)
            )
        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.segmentation_classes = segmentation_classes

    def get_random_transformation_batch(self, batch_size, **kwargs):
        min_angle = self.lower * 2.0 * np.pi
        max_angle = self.upper * 2.0 * np.pi
        angles = self._random_generator.uniform(
            shape=[batch_size], minval=min_angle, maxval=max_angle
        )
        return {"angles": angles}

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = {
            "angles": tf.expand_dims(transformation["angles"], axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        return self._rotate_images(images, transformations)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomRotation()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomRotation(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)

        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )
        image_shape = tf.shape(raw_images)
        h = image_shape[H_AXIS]
        w = image_shape[W_AXIS]

        # origin coordinates, all the points on the image are rotated around
        # this point
        origin_x = tf.cast(w / 2, dtype=self.compute_dtype)
        origin_y = tf.cast(h / 2, dtype=self.compute_dtype)
        angles = -transformations["angles"]
        angles = angles[:, tf.newaxis, tf.newaxis, tf.newaxis]

        # calculate coordinates of all four corners of the bounding box
        boxes = bounding_boxes["boxes"]
        points = tf.stack(
            [
                tf.stack([boxes[:, :, 0], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 3]], axis=2),
                tf.stack([boxes[:, :, 0], boxes[:, :, 3]], axis=2),
            ],
            axis=2,
        )
        # point_x : x coordinates of all corners of the bounding box
        point_xs = tf.gather(points, [0], axis=3)
        point_x_offsets = tf.cast((point_xs - origin_x), dtype=tf.float32)
        # point_y : y coordinates of all corners of the bounding box
        point_ys = tf.gather(points, [1], axis=3)
        point_y_offsets = tf.cast((point_ys - origin_y), dtype=tf.float32)
        # rotated bounding box coordinates
        # new_x : new position of x coordinates of corners of bounding box
        new_x = (
            origin_x
            + tf.multiply(tf.cos(angles), point_x_offsets)
            - tf.multiply(tf.sin(angles), point_y_offsets)
        )
        # new_y : new position of y coordinates of corners of bounding box
        new_y = (
            origin_y
            + tf.multiply(tf.sin(angles), point_x_offsets)
            + tf.multiply(tf.cos(angles), point_y_offsets)
        )
        # rotated bounding box coordinates
        out = tf.concat([new_x, new_y], axis=3)
        # find readjusted coordinates of bounding box to represent it in corners
        # format
        min_coordinates = tf.math.reduce_min(out, axis=2)
        max_coordinates = tf.math.reduce_max(out, axis=2)
        boxes = tf.concat([min_coordinates, max_coordinates], axis=2)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=raw_images,
        )
        # coordinates cannot be float values, it is cast to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        # If segmentation_classes is specified, we have a dense segmentation
        # mask. We therefore one-hot encode before rotation to avoid bad
        # interpolation during the rotation transformation. We then make the
        # mask sparse again using tf.argmax.
        if self.segmentation_classes:
            one_hot_mask = tf.one_hot(
                tf.squeeze(tf.cast(segmentation_masks, tf.int32), axis=-1),
                self.segmentation_classes,
            )
            rotated_one_hot_mask = self._rotate_images(
                one_hot_mask, transformations
            )
            rotated_mask = tf.argmax(rotated_one_hot_mask, axis=-1)
            return tf.expand_dims(rotated_mask, axis=-1)
        else:
            if segmentation_masks.shape[-1] == 1:
                raise ValueError(
                    "Segmentation masks must be one-hot encoded, or "
                    "RandomRotate must be initialized with "
                    "`segmentation_classes`. `segmentation_classes` was not "
                    f"specified, and mask has shape {segmentation_masks.shape}"
                )
            rotated_mask = self._rotate_images(
                segmentation_masks, transformations
            )
            # Round because we are in one-hot encoding, and we may have
            # pixels with ambiguous value due to floating point math for
            # rotation.
            return tf.round(rotated_mask)

    def _rotate_images(self, images, transformations):
        images = preprocessing_utils.ensure_tensor(images, self.compute_dtype)
        original_shape = images.shape
        image_shape = tf.shape(images)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        angles = transformations["angles"]
        outputs = preprocessing_utils.transform(
            images,
            preprocessing_utils.get_rotation_matrix(angles, img_hd, img_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        outputs.set_shape(original_shape)
        return outputs

    def get_config(self):
        config = {
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "bounding_box_format": self.bounding_box_format,
            "segmentation_classes": self.segmentation_classes,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
