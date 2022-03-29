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

from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Equalization(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Equalization performs histogram equalization on a channel-wise basis.

    Args:
        bins: Integer indicating the number of bins to use in histogram equalization.
            Should be in the range [0, 256]
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`. Defaults to `(0, 255)`.

    Usage:
    ```python
    equalize = Equalization()

    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    # Note that images are an int8 Tensor with values in the range [0, 255]
    images = equalize(images)
    ```

    Call arguments:
        images: Tensor of pixels in range [0, 255], in RGB format.  Can be
            of type float or int.  Should be in NHWC format.
    """

    def __init__(self, bins=256, value_range=(0, 255), **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.value_range = value_range

    def equalize_channel(self, image, channel_index):
        """equalize_channel performs histogram equalization on a single channel.

        Args:
            image: int Tensor with pixels in range [0, 255], RGB format,
                with channels last
            channel_index: channel to equalize
        """
        image = image[..., channel_index]
        # Compute the histogram of the image channel.
        histogram = tf.histogram_fixed_width(image, [0, 255], nbins=self.bins)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histogram, 0))
        nonzero_histogram = tf.reshape(tf.gather(histogram, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histogram) - nonzero_histogram[-1]) // (
            self.bins - 1
        )

        def build_mapping(histogram, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lookup_table = (tf.cumsum(histogram) + (step // 2)) // step
            # Shift lookup_table, prepending with 0.
            lookup_table = tf.concat([[0], lookup_table[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lookup_table, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lookup table from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0),
            lambda: image,
            lambda: tf.cast(
                tf.gather(build_mapping(histogram, step), tf.cast(image, tf.int32)),
                self.compute_dtype,
            ),
        )

        return result

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(image, self.value_range, (0, 255))
        r = self.equalize_channel(image, 0)
        g = self.equalize_channel(image, 1)
        b = self.equalize_channel(image, 2)
        image = tf.stack([r, g, b], axis=-1)
        image = preprocessing.transform_value_range(image, (0, 255), self.value_range)
        return image

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins, "value_range": self.value_range})
        return config
