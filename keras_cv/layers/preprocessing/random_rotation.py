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

import functools

import numpy as np
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import keypoint

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing

# In order to support both unbatched and batched inputs, the horizontal
# and verticle axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


class MissingFormatError:
    @staticmethod
    def keypoints(cls_name):
        return ValueError(
            f"`{cls_name}()` was called with keypoints, but no `keypoint_format` was "
            "specified in the constructor. Please specify a keypoint format in the "
            f"constructor, i.e. `{cls_name}(keypoint_format='xy')"
        )

    @staticmethod
    def bounding_boxes(cls_name):
        return ValueError(
            f"`{cls_name}()` was called with bounding boxes, but no ",
            "`bounding_box_format` was specified in the constructor. Please specify a "
            "bounding box format in the constructor, i.e. "
            f"`{cls_name}(keypoint_format='xy')"
        )


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomRotation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly rotates images during training.

    This layer will apply random rotations to each image, filling empty space
    according to `fill_mode`.

    By default, random rotations are only applied during training.
    At inference time, the layer does nothing. If you need to apply random
    rotations at inference time, set `training` to True when calling the layer.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output
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
      keypoint_format: The format of keypoints of input dataset. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported keypoint formats.
      clip_points_to_image_size: Indicates if bounding boxes and
        keypoints should respectively be clipped or discarded
        according to the input image shape. Note that you should
        disable clipping while augmenting batches of images with
        keypoints data.

    """
    def __init__(
        self,
        factor,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
        bounding_box_format=None,
        keypoint_format=None,
        clip_points_to_image_size=True,
        **kwargs,
    ):
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.factor = factor
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError(
                "Factor cannot have negative values, "
                "got {}".format(factor)
            )
        preprocessing.check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self.bounding_box_format = bounding_box_format
        self.keypoint_format = keypoint_format
        self.clip_points_to_image_size = clip_points_to_image_size

    def get_random_transformation(self, **kwargs):
        min_angle = self.lower * 2.0 * np.pi
        max_angle = self.upper * 2.0 * np.pi
        angle = self._random_generator.random_uniform(
            shape=[1], minval=min_angle, maxval=max_angle
        )
        return {"angle": angle}

    def augment_image(self, image, transformation, **kwargs):
        image = preprocessing.ensure_tensor(image, self.compute_dtype)
        original_shape = image.shape
        image = tf.expand_dims(image, 0)
        image_shape = tf.shape(image)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        angle = transformation["angle"]
        output = preprocessing.transform(
            image,
            preprocessing.get_rotation_matrix(angle, img_hd, img_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        output = tf.squeeze(output, 0)
        output.set_shape(original_shape)
        return output

    def augment_bounding_boxes(self, bounding_boxes, transformation, **kwargs):
        image = None
        image = kwargs.get("image")
        if self.bounding_box_format is None:
            raise MissingFormatError.bounding_boxes("RandomRotation")
        # calculate bounding box by applying keypoints transformation on the bounding box corners
        return bounding_box.transform_from_point_transform(
            bounding_boxes,
            functools.partial(
                self.augment_keypoints,
                transformation=transformation,
                image=image,
                keypoint_format='xy',
                discard_out_of_image=False,
            ),
            bounding_box_format=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=image,
            clip_boxes=self.clip_points_to_image_size,
        )

    def augment_keypoints(
        self,
        keypoints,
        transformation,
        image=None,
        **kwargs,
    ):

        keypoint_format = kwargs.get("keypoint_format", self.keypoint_format)
        discard_out_of_image = kwargs.get(
            "discard_out_of_image", self.clip_points_to_image_size
        )
        if keypoint_format is None:
            raise MissingFormatError.keypoints("RandomRotation")

        keypoints = keypoint.convert_format(
            keypoints, source=keypoint_format, target='xy', images=image
        )

        image = tf.expand_dims(image, 0)
        image_shape = tf.shape(image)
        h = image_shape[H_AXIS]
        w = image_shape[W_AXIS]
        # origin coordinates, all the points on the image are rotated around
        # this point
        origin_x, origin_y = tf.cast(h // 2, tf.float32), tf.cast(w // 2, tf.float32)
        angle = transformation["angle"]
        angle = -angle
        # point_x : x coordinates of all corners of the bounding box
        point_x = tf.gather(keypoints, [0], axis=-1)
        # point_y : y cordinates of all corners of the bounding box
        point_y = tf.gather(keypoints, [1], axis=-1)
        # rotated bounding box coordinates
        # new_x : new position of x coordinates of corners of bounding box
        new_x = (
            origin_x +
            tf.multiply(tf.cos(angle), tf.cast((point_x - origin_x),
                                               dtype=tf.float32)) -
            tf.multiply(tf.sin(angle), tf.cast((point_y - origin_y), dtype=tf.float32))
        )
        # new_y : new position of y coordinates of corners of bounding box
        new_y = (
            origin_y +
            tf.multiply(tf.sin(angle), tf.cast((point_x - origin_x),
                                               dtype=tf.float32)) +
            tf.multiply(tf.cos(angle), tf.cast((point_y - origin_y), dtype=tf.float32))
        )
        # rotated bounding box coordinates
        out = tf.concat([new_x, new_y], axis=-1)

        if discard_out_of_image:
            out = keypoint.discard_out_of_image(out, image)

        return keypoint.convert_format(
            out, source='xy', target=keypoint_format, images=image
        )

    def augment_label(self, label, transformation, **kwargs):
        return label

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "factor": self.factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "clip_points_to_image_size": self.clip_points_to_image_size,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
