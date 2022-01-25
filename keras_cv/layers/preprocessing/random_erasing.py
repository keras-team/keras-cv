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
import tensorflow.keras.layers as layers
from tensorflow.keras import backend
from tensorflow.python.keras.utils import layer_utils

from keras_cv.utils import fill_utils


class RandomErasing(layers.Layer):
    """RandomErasing implements the RandomErasing data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        area_factor: a positive float represented as fraction of image area, or
            a tuple of size 2 representing lower and upper bound for patch area
            relative to image area. When represented as a single positive float,
            lower = upper. For instance, `area_factor=(0.2, 0.3)` results in a
            patch area randomly picked in the range
            `[20% of image area, 30% of image area]`. `area_factor=0.2` results
            in a patch area of 20% of image area.
        aspect_ratio: a positive float represented as aspect ratio, or a tuple of
            size 2 representing lower and upper bound for patch aspect ratio. When
            represented as a single positive float, lower = upper. For instance,
            `aspect_ratio=(0.3, 3.3)` results in a patch with aspect ratio randomly
            picked in the range `[0.3, 3.3]`. `aspect_ratio=0.2` results in a patch
            with aspect ratio of 0.2.
        fill_mode: Pixels inside the patches are filled according to the given
            mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: a float represents the value to be filled inside the patches
          when `fill_mode="constant"`.
    References:
       [RandomErasing paper](https://arxiv.org/abs/1708.04896).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    random_erase = keras_cv.layers.preprocessing.erasing.RandomErasing(1.0)
    augmented_images = random_erase(images)
    ```
    """

    def __init__(
        self,
        rate,
        area_factor=(0.02, 0.33),
        aspect_ratio=(0.3, 3.3),
        fill_mode="gaussian_noise",
        fill_value=0.0,
        seed=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        layer_utils.validate_string_arg(
            fill_mode,
            allowable_strings=["constant", "gaussian_noise"],
            layer_name="RandomErasing",
            arg_name="fill_mode",
            allow_none=False,
            allow_callables=False,
        )

        if isinstance(area_factor, (tuple, list)):
            self.area_factor = area_factor
        else:
            self.area_factor = (area_factor, area_factor)

        if isinstance(aspect_ratio, (tuple, list)):
            self.aspect_ratio = aspect_ratio
        else:
            self.aspect_ratio = (aspect_ratio, aspect_ratio)

        self.rate = rate
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.seed = seed

    def call(self, images, training=True):
        """call method for the layer.

        Args:
            images: Tensor representing images of shape
            [batch_size, width, height, channels], with dtype tf.float32.
        Returns:
            images: augmented images, same shape as input.
        """
        if training is None:
            training = backend.learning_phase()

        rate_cond = tf.less(
            tf.random.uniform(shape=[], minval=0.0, maxval=1.0), self.rate
        )
        augment_cond = tf.logical_and(rate_cond, training)
        # pylint: disable=g-long-lambda
        augment = lambda: self._random_erase(images)
        no_augment = lambda: images
        return tf.cond(augment_cond, augment, no_augment)

    def _random_erase(self, images):
        """Apply random erasing."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        patch_height, patch_width = self._compute_patch_size(
            batch_size, image_height, image_width
        )

        random_center_height = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
            seed=self.seed,
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
            seed=self.seed,
        )

        args = [
            images,
            random_center_width,
            random_center_height,
            patch_width // 2,
            patch_height // 2,
        ]
        if self.fill_mode == "constant":
            patch_value = tf.fill([batch_size], self.fill_value)
            args.append(patch_value)

        images = tf.map_fn(
            lambda x: fill_utils.fill_rectangle(*x),
            args,
            fn_output_signature=tf.TensorSpec.from_tensor(images[0]),
        )

        return images

    def _compute_patch_size(self, batch_size, image_height, image_width):
        area = tf.cast(image_height * image_width, tf.float32)
        erase_area = area * tf.random.uniform(
            [batch_size], minval=self.area_factor[0], maxval=self.area_factor[1]
        )
        aspect_ratio = tf.random.uniform(
            [batch_size], minval=self.aspect_ratio[0], maxval=self.aspect_ratio[1]
        )
        h = tf.cast(tf.round(tf.sqrt(erase_area * aspect_ratio)), tf.int32)
        w = tf.cast(tf.round(tf.sqrt(erase_area / aspect_ratio)), tf.int32)

        h = tf.minimum(h, image_height - 1)
        w = tf.minimum(w, image_width - 1)

        return h, w

    def get_config(self):
        config = {
            "rate": self.rate,
            "area_factor": self.area_factor,
            "aspect_ratio": self.aspect_ratio,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
