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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.RandomGaussianBlur")
class RandomGaussianBlur(BaseImageAugmentationLayer):
    """Applies a Gaussian Blur with random strength to an image.

    Args:
        kernel_size: int, 2 element tuple or 2 element list. x and y dimensions
            for the kernel used. If tuple or list, first element is used for the
            x dimension and second element is used for y dimension. If int,
            kernel will be squared.
        factor: A tuple of two floats, a single float or a
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image is blurred. Mathematically, `factor` represents the `sigma`
            value in a gaussian blur. `factor=0.0` makes this layer perform a
            no-op operation, and high values make the blur stronger. In order to
            ensure the value is always the same, please pass a tuple with two
            identical floats: `(0.5, 0.5)`.
    """

    def __init__(self, kernel_size, factor, **kwargs):
        super().__init__(**kwargs)

        self.factor = preprocessing.parse_factor(
            factor, min_value=0.0, max_value=None, param_name="factor"
        )

        self.kernel_size = kernel_size

        if isinstance(kernel_size, (tuple, list)):
            self.x = kernel_size[0]
            self.y = kernel_size[1]
        else:
            if isinstance(kernel_size, int):
                self.x = self.y = kernel_size
            else:
                raise ValueError(
                    "`kernel_size` must be list, tuple or integer "
                    ", got {} ".format(type(self.kernel_size))
                )

    def get_random_transformation(self, **kwargs):
        # `factor` must not become too small otherwise numerical issues occur.
        # keras.backend.epsilon() behaves like 0 without causing `nan`s
        factor = tf.math.maximum(self.factor(), keras.backend.epsilon())
        blur_v = RandomGaussianBlur.get_kernel(factor, self.y)
        blur_h = RandomGaussianBlur.get_kernel(factor, self.x)
        blur_v = tf.reshape(blur_v, [self.y, 1, 1, 1])
        blur_h = tf.reshape(blur_h, [1, self.x, 1, 1])
        return (blur_v, blur_h)

    def augment_image(self, image, transformation=None, **kwargs):
        image = tf.expand_dims(image, axis=0)

        num_channels = tf.shape(image)[-1]
        blur_v, blur_h = transformation
        blur_h = tf.cast(
            tf.tile(blur_h, [1, 1, num_channels, 1]), dtype=self.compute_dtype
        )
        blur_v = tf.cast(
            tf.tile(blur_v, [1, 1, num_channels, 1]), dtype=self.compute_dtype
        )
        blurred = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )

        return tf.squeeze(blurred, axis=0)

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def augment_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        return segmentation_mask

    @staticmethod
    def get_kernel(factor, filter_size):
        # We are running this in float32, regardless of layer's
        # self.compute_dtype. Calculating blur_filter in lower precision will
        # corrupt the final results.
        x = tf.cast(
            tf.range(-filter_size // 2 + 1, filter_size // 2 + 1),
            dtype=tf.float32,
        )
        blur_filter = tf.exp(
            -tf.pow(x, 2.0)
            / (2.0 * tf.pow(tf.cast(factor, dtype=tf.float32), 2.0))
        )
        blur_filter /= tf.reduce_sum(blur_filter)
        return blur_filter

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "kernel_size": self.kernel_size})
        return config
