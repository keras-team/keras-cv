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

from keras_cv.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)
from keras_cv.utils import preprocessing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Equalization(BaseImageAugmentationLayer):
    """Equalization performs histogram equalization on a channel-wise basis.

    Args:
        value_range: a tuple or a list of two elements. The first value represents
            the lower bound for values in passed images, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`.
        bins: Integer indicating the number of bins to use in histogram equalization.
            Should be in the range [0, 256].

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

    def __init__(self, value_range, bins=256, **kwargs):
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
        # Zeroes are replaced by a big number while calculating min to keep shape
        # constant across input sizes for compatibility with vectorized_map

        big_number = 1410065408
        histogram_without_zeroes = tf.where(
            tf.equal(histogram, 0),
            big_number,
            histogram,
        )

        step = (tf.reduce_sum(histogram) - tf.reduce_min(histogram_without_zeroes)) // (
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
            lambda: tf.gather(build_mapping(histogram, step), image),
        )

        return result

    def augment_image(self, image, transformation=None, **kwargs):
        image = preprocessing.transform_value_range(
            image, self.value_range, (0, 255), dtype=image.dtype
        )
        image = tf.cast(image, tf.int32)
        image = tf.map_fn(
            lambda channel: self.equalize_channel(image, channel),
            tf.range(tf.shape(image)[-1]),
        )

        image = tf.transpose(image, [1, 2, 0])
        image = tf.cast(image, tf.float32)
        image = preprocessing.transform_value_range(image, (0, 255), self.value_range)
        return image

    def augment_label(self, label, transformation=None, **kwargs):
        return label

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins, "value_range": self.value_range})
        return config
