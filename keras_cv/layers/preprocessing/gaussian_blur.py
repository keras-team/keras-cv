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
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma

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
