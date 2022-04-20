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
<<<<<<< HEAD

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GaussianBlur(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Applies a Gaussian Blur with random sigma to an image.

    Args:
        kernel_size: int, 2 element tuple or 2 element list. x and y dimensions for
            the kernel used. If tuple or list, first element is used for the x dimension
            and second element is used for y dimension. If int, kernel will be squared.
        sigma: float, 2 element tuple or 2 element list. Interval in which sigma should
            be sampled from. If float, interval is going to be [0, float), else the
            first element represents the lower bound and the second element the upper
            bound of the sampling interval.
    """

    def __init__(self, kernel_size, sigma, **kwargs):
=======
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend
from tensorflow.keras import layers


class GaussianBlur(layers.Layer):
    """GaussianBlur is a preprocessing layer that applies Gaussian Blur to RGB and
    greyscale images. Input images should have values in the range of [0,255]

        Args:
            kernel_size: An integer or tuple/list of 2 integers, specifying
                height and weight of Gaussian kernel. If integer, this represents
                both dimensions of a square kernel.
            sigma: Float or tuple/list of 2 floats representing the standard deviation
                used to calculate kernel. tuple/list can be used to apply different
                standard deviations for height and weight.

        Usage:
        ```python
        blur = GaussianBlur()

        (images, labels), _ = tf.keras.datasets.cifar10.load_data()
        # Note that images are an int8 Tensor with values in the range [0, 255]
        images = equalize(images)
        ```

        Call arguments:
            images: Tensor of pixels in range [0, 255], in RGB or greyscale format.
                Can be of type float or int.  Should be in NHWC format.
    """

    def __init__(self, kernel_size=5, sigma=1, **kwargs):
>>>>>>> 93f10db9b94d51cf814b39d3b17e358bc2ecdd3d
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

<<<<<<< HEAD
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

        if isinstance(sigma, (tuple, list)):
            self.sigma_min = sigma[0]
            self.sigma_max = sigma[1]
        else:
            self.sigma_min = type(sigma)(0)
            self.sigma_max = sigma

        if not isinstance(self.sigma_min, type(self.sigma_max)):
            raise ValueError(
                "`sigma` must have lower bound and upper bound "
                "with same type, got {} and {}".format(
                    type(self.width_lower), type(self.width_upper)
                )
            )

        if self.sigma_max < self.sigma_min:
            raise ValueError(
                "`sigma` cannot have upper bound less than "
                "lower bound, got {}".format(sigma)
            )

        self._sigma_is_float = isinstance(self.sigma, float)
        if self._sigma_is_float:
            if not self.sigma_min >= 0.0:
                raise ValueError(
                    "`sigma` must be higher than 0"
                    "when is float, got {}".format(sigma)
                )

    def get_random_transformation(self, image=None, label=None, bounding_box=None):
        sigma = self.get_sigma()
        blur_v = GaussianBlur.get_kernel(sigma, self.y)
        blur_h = GaussianBlur.get_kernel(sigma, self.x)
        blur_v = tf.reshape(blur_v, [self.y, 1, 1, 1])
        blur_h = tf.reshape(blur_h, [1, self.x, 1, 1])
        return (blur_v, blur_h)

    def get_sigma(self):
        sigma = self._random_generator.random_uniform(
            shape=(), minval=self.sigma_min, maxval=self.sigma_max
        )
        return sigma

    def augment_image(self, image, transformation=None):

        image = tf.expand_dims(image, axis=0)

        num_channels = tf.shape(image)[-1]
        blur_v, blur_h = transformation
        blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
        blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
        blurred = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        blurred = tf.nn.depthwise_conv2d(
            blurred, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )

        return tf.squeeze(blurred, axis=0)

    @staticmethod
    def get_kernel(sigma, filter_size):
        x = tf.cast(
            tf.range(-filter_size // 2 + 1, filter_size // 2 + 1), dtype=tf.float32
        )
        blur_filter = tf.exp(
            -tf.pow(x, 2.0) / (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0))
        )
        blur_filter /= tf.reduce_sum(blur_filter)
        return blur_filter

    def get_config(self):
        config = super().get_config()
        config.update({"sigma": self.sigma, "kernel_size": self.kernel_size})
        return config
=======
    def call(self, inputs, training=True):
        """call method for the GaussianBlur layer.
        Args:
            images: Tensor representing images of shape
                [batch_size, width, height, channels] or
                [width, height, channels] with type float or int.
                Pixel values should be in the range [0, 255]
        Returns:
            images: Blurred input images, same as input.
        """
        if training is None:
            training = backend.learning_phase()

        def _blur(image, kernel_size, sigma):
            image = tfa.image.gaussian_filter2d(
                image=image,
                filter_shape=kernel_size,
                sigma=sigma,
                padding="CONSTANT",
                constant_values=1,
            )
            return image

        augment = lambda: _blur(inputs, self.kernel_size, self.sigma)
        no_augment = lambda: inputs
        return tf.cond(tf.cast(training, tf.bool), augment, no_augment)

    def get_config(self):
        config = {"kernel_size": self.kernel_size, "sigma": self.sigma}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
>>>>>>> 93f10db9b94d51cf814b39d3b17e358bc2ecdd3d
