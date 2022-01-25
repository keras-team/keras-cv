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


class CutOut(layers.Layer):
    """CutOut implements the CutOut data augmentation technique.

    Args:
        rate: Float between 0 and 1.  The fraction of samples to augment.
        length: Integer. The side length of the square patches to cut out.
        fill_mode: Pixels inside the patches are filled according
            to the given mode (one of `{"constant", "gaussian_noise"}`).
            - *constant*: Pixels are filled with the same constant value.
            - *gaussian_noise*: Pixels are filled with random gaussian noise.
        fill_value: a float represents the value to be filled inside the patches
          when `fill_mode="constant"`.
    References:
       [CutOut paper](https://arxiv.org/abs/1708.04552).

    Sample usage:
    ```python
    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    cutout = keras_cv.layers.preprocessing.erasing.CutOut(1.0, 50)
    augmented_images = cutout(images)
    ```
    """

    def __init__(
        self,
        rate,
        length,
        fill_mode="constant",
        fill_value=0.0,
        seed=None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        layer_utils.validate_string_arg(
            fill_mode,
            allowable_strings=["constant", "gaussian_noise"],
            layer_name="CutOut",
            arg_name="fill_mode",
            allow_none=False,
            allow_callables=False,
        )

        self.rate = rate
        self.length = length
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
        augment = lambda: self._cutout(images)
        no_augment = lambda: images
        return tf.cond(augment_cond, augment, no_augment)

    def _cutout(self, images):
        """Apply cut out."""
        input_shape = tf.shape(images)
        batch_size, image_height, image_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )

        patch_height = tf.fill([batch_size], self.length)
        patch_width = patch_height

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

    def get_config(self):
        config = {
            "rate": self.rate,
            "length": self.length,
            "fill_mode": self.fill_mode,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
