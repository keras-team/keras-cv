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

import tensorflow as tf

from keras_cv import bounding_box
from keras_cv import keypoint
from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.layers.preprocessing.random_rotation import MissingFormatError

H_AXIS = -3
W_AXIS = -2


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomTranslation(BaseImageAugmentationLayer):
    """A preprocessing layer which randomly translates images during training.

    This layer will apply random translations to each image during training,
    filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype. By default, the layer will output floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
      height_factor: oa float represented as fraction of value, or a
        tuple of size 2 representing lower and upper bound for
        shifting vertically. A negative value means shifting image up,
        while a positive value means shifting image down. When
        represented as a single positive float, this value is used for
        both the upper and lower bound. For instance,
        `height_factor=(-0.2, 0.3)` results in an output shifted by a
        random amount in the range `[-20%, +30%]`.
        `height_factor=0.2` results in an output height shifted by a
        random amount in the range `[-20%, +20%]`.
      width_factor: a float represented as fraction of value, or a
        tuple of size 2 representing lower and upper bound for
        shifting horizontally. A negative value means shifting image
        left, while a positive value means shifting image right. When
        represented as a single positive float, this value is used for
        both the upper and lower bound. For instance,
        `width_factor=(-0.2, 0.3)` results in an output shifted left
        by 20%, and shifted right by 30%. `width_factor=0.2` results
        in an output height shifted left or right by 20%.
      fill_mode: Points outside the boundaries of the input are filled according
        to the given mode (one of `{"constant", "reflect", "wrap", "nearest"}`).
        - *reflect*: `(d c b a | a b c d | d c b a)` The input is
          extended by reflecting about the edge of the last pixel.
        - *constant*: `(k k k k | a b c d | k k k k)` The input is extended by
          filling all values beyond the edge with the same constant value k = 0.
        - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
          wrapping around to the opposite edge.
        - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended by the
          nearest pixel.
      interpolation: Interpolation mode. Supported values: `"nearest"`,
        `"bilinear"`.
      bounding_box_format: The format of bounding boxes of input
        dataset. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported bounding box formats.
      keypoint_format: The format of keypoints of input dataset. Refer
        to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
        for more details on supported keypoint formats.
      clip_points_to_image_size: Indicates if bounding boxes and
        keypoints should respectively be clipped or discarded
        according to the input image shape. Note that you should
        disable clipping while augmenting batches of images with
        keypoints data.
      seed: Integer. Used to create a random seed.
      fill_value: a float represents the value to be filled outside the boundaries
        when `fill_mode="constant"`.
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
        keypoint_format=None,
        clip_points_to_image_size=True,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.base = tf.keras.layers.RandomTranslation(
            height_factor=height_factor,
            width_factor=width_factor,
            fill_mode=fill_mode,
            interpolation=interpolation,
            seed=seed,
            fill_value=fill_value,
            **kwargs,
        )
        self.bounding_box_format = bounding_box_format
        self.keypoint_format = keypoint_format
        self.clip_points_to_image_size = clip_points_to_image_size

    def get_config(self):
        config = super().get_config()
        config.update(self.base.get_config())
        config.update(
            {
                "bounding_box_format": self.bounding_box_format,
                "keypoint_format": self.keypoint_format,
                "clip_points_to_image_size": self.clip_points_to_image_size,
            }
        )
        return config

    def get_random_transformation(
        self, image=None, label=None, bounding_boxes=None, **kwargs
    ):
        return self.base.get_random_transformation(
            image=image, label=label, bounding_box=bounding_boxes
        )

    def augment_image(self, image, transformation=None, **kwargs):
        return self.base.augment_image(image=image, transformation=transformation)

    def augment_label(self, labels, transformation=None, **kwargs):
        return labels

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformation=None,
        image=None,
        **kwargs,
    ):
        if self.bounding_box_format is None:
            raise MissingFormatError.bounding_boxes("RandomTranslation")

        return bounding_box.transform_from_point_transform(
            bounding_boxes,
            functools.partial(
                self.augment_keypoints,
                transformation=transformation,
                image=image,
                keypoint_format="xy",
                discard_out_of_image=False,
            ),
            bounding_box_format=self.bounding_box_format,
            images=image,
            dtype=self.compute_dtype,
            clip_boxes=self.clip_points_to_image_size,
        )

    def augment_keypoints(
        self,
        keypoints,
        transformation=None,
        **kwargs,
    ):
        image = kwargs.get("image")
        discard_out_of_image = kwargs.get(
            "discard_out_of_image", self.clip_points_to_image_size
        )

        keypoint_format = kwargs.get("keypoint_format", self.keypoint_format)

        if keypoint_format is None:
            raise MissingFormatError.keypoints("RandomTranslation")

        keypoints = keypoint.convert_format(
            keypoints,
            source=keypoint_format,
            target="xy",
            images=image,
        )

        inputs_shape = tf.shape(image)
        img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)

        offset_x = transformation["width_translation"]
        offset_y = transformation["height_translation"]

        offset = tf.cast(
            tf.concat([img_wd * offset_x, img_hd * offset_y], axis=-1),
            dtype=self.compute_dtype,
        )

        out = keypoints + offset

        if discard_out_of_image:
            out = keypoint.discard_out_of_image(out, image=image)

        return keypoint.convert_format(
            out, source="xy", target=keypoint_format, images=image
        )
