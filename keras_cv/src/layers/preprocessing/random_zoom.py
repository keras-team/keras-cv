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
from tensorflow.keras import backend

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing as preprocessing_utils

# In order to support both unbatched and batched inputs, the horizontal
# and vertical axis is reverse indexed
H_AXIS = -3
W_AXIS = -2


@keras_cv_export("keras_cv.layers.RandomZoom")
class RandomZoom(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly zooms images.

    This layer will randomly zoom in or out on each axis of an image
    independently, filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype. By default, the layer will output
    floats.

    Args:
      height_factor: a float represented as fraction of value, or a tuple of
        size 2 representing lower and upper bound for zooming vertically. When
        represented as a single float, this value is used for both the upper and
        lower bound. A positive value means zooming out, while a negative value
        means zooming in. For instance, `height_factor=(0.2, 0.3)` result in an
        output zoomed out by a random amount in the range `[+20%, +30%]`.
        `height_factor=(-0.3, -0.2)` result in an output zoomed in by a random
        amount in the range `[-30%, -20%]`.
      width_factor: a float represented as fraction of value, or a tuple of size
        2 representing lower and upper bound for zooming horizontally. When
        represented as a single float, this value is used for both the upper and
        lower bound. For instance, `width_factor=(0.2, 0.3)` result in an output
        zooming out between 20% to 30%. `width_factor=(-0.3, -0.2)` result in an
        output zooming in between 20% to 30%. Defaults to `None`, i.e., zooming
        vertical and horizontal directions by preserving the aspect ratio. If
        height_factor=0 and width_factor=None, it would result in images with
        no zoom at all.
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

    Example:

    >>> input_img = np.random.random((32, 224, 224, 3))
    >>> layer = keras_cv.layers.RandomZoom(.5, .2)
    >>> out_img = layer(input_img)
    >>> out_img.shape
    TensorShape([32, 224, 224, 3])

    Input shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
      3D (unbatched) or 4D (batched) tensor with shape:
      `(..., height, width, channels)`, in `"channels_last"` format.
    """

    def __init__(
        self,
        height_factor,
        width_factor=None,
        fill_mode="reflect",
        interpolation="bilinear",
        seed=None,
        fill_value=0.0,
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

        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(
                "`height_factor` must have values between [-1, 1], "
                f"got {height_factor}"
            )

        self.width_factor = width_factor
        if width_factor is not None:
            if isinstance(width_factor, (tuple, list)):
                self.width_lower = width_factor[0]
                self.width_upper = width_factor[1]
            else:
                self.width_lower = -width_factor
                self.width_upper = width_factor

            if self.width_lower < -1.0 or self.width_upper < -1.0:
                raise ValueError(
                    "`width_factor` must have values larger than -1, "
                    f"got {width_factor}"
                )

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        height_zooms = self._random_generator.uniform(
            shape=[batch_size, 1],
            minval=1.0 + self.height_lower,
            maxval=1.0 + self.height_upper,
        )
        if self.width_factor is not None:
            width_zooms = self._random_generator.uniform(
                shape=[batch_size, 1],
                minval=1.0 + self.width_lower,
                maxval=1.0 + self.width_upper,
            )
        else:
            width_zooms = height_zooms

        return {"height_zooms": height_zooms, "width_zooms": width_zooms}

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        width_zooms = transformation["width_zooms"]
        height_zooms = transformation["height_zooms"]
        transformation = {
            "height_zooms": tf.expand_dims(height_zooms, axis=0),
            "width_zooms": tf.expand_dims(width_zooms, axis=0),
        }
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.ensure_tensor(images, self.compute_dtype)
        original_shape = images.shape
        image_shape = tf.shape(images)
        img_hd = tf.cast(image_shape[H_AXIS], tf.float32)
        img_wd = tf.cast(image_shape[W_AXIS], tf.float32)
        width_zooms = transformations["width_zooms"]
        height_zooms = transformations["height_zooms"]
        zooms = tf.cast(
            tf.concat([width_zooms, height_zooms], axis=1), dtype=tf.float32
        )
        outputs = preprocessing_utils.transform(
            images,
            self.get_zoom_matrix(zooms, img_hd, img_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        outputs.set_shape(original_shape)
        return outputs

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
        mask_hd = tf.cast(mask_shape[H_AXIS], tf.float32)
        mask_wd = tf.cast(mask_shape[W_AXIS], tf.float32)
        width_zooms = transformations["width_zooms"]
        height_zooms = transformations["height_zooms"]
        zooms = tf.cast(
            tf.concat([width_zooms, height_zooms], axis=1), dtype=tf.float32
        )
        outputs = preprocessing_utils.transform(
            segmentation_masks,
            self.get_zoom_matrix(zooms, mask_hd, mask_wd),
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation="nearest",
        )
        outputs.set_shape(original_shape)
        return outputs

    def get_zoom_matrix(self, zooms, image_height, image_width, name=None):
        """Returns projective transform(s) for the given zoom(s).

        Args:
        zooms: A matrix of 2-element lists representing `[zx, zy]` to zoom for
            each image (for a batch of images).
        image_height: Height of the image(s) to be transformed.
        image_width: Width of the image(s) to be transformed.
        name: The name of the op.

        Returns:
        A tensor of shape `(num_images, 8)`. Projective transforms which can be
            given to operation `image_projective_transform_v2`.
            If one row of transforms is
            `[a0, a1, a2, b0, b1, b2, c0, c1]`, then it maps the *output* point
            `(x, y)` to a transformed *input* point
            `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
            where `k = c0 x + c1 y + 1`.
        """
        with backend.name_scope(name or "zoom_matrix"):
            num_zooms = tf.shape(zooms)[0]
            # The zoom matrix looks like:
            #     [[zx 0 0]
            #      [0 zy 0]
            #      [0 0 1]]
            # where the last entry is implicit.
            # Zoom matrices are always float32.
            x_offset = ((image_width - 1.0) / 2.0) * (1.0 - zooms[:, 0, None])
            y_offset = ((image_height - 1.0) / 2.0) * (1.0 - zooms[:, 1, None])
            return tf.concat(
                values=[
                    zooms[:, 0, None],
                    tf.zeros((num_zooms, 1), tf.float32),
                    x_offset,
                    tf.zeros((num_zooms, 1), tf.float32),
                    zooms[:, 1, None],
                    y_offset,
                    tf.zeros((num_zooms, 2), tf.float32),
                ],
                axis=1,
            )

    def get_config(self):
        config = {
            "height_factor": self.height_factor,
            "width_factor": self.width_factor,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "interpolation": self.interpolation,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
