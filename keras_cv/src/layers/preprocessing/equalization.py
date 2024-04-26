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

from functools import partial

import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from keras_cv.src.utils import preprocessing


@keras_cv_export("keras_cv.layers.Equalization")
class Equalization(VectorizedBaseImageAugmentationLayer):
    """Equalization performs histogram equalization on a channel-wise basis.

    Args:
        value_range: a tuple or a list of two elements. The first value
            represents the lower bound for values in passed images, the second
            represents the upper bound. Images passed to the layer should have
            values within `value_range`.
        bins: Integer indicating the number of bins to use in histogram
            equalization. Should be in the range [0, 256].

    Example:
    ```python
    equalize = Equalization()

    (images, labels), _ = keras.datasets.cifar10.load_data()
    # Note that images are an int8 Tensor with values in the range [0, 255]
    images = equalize(images)
    ```

    Call arguments:
        images: Tensor of pixels in range [0, 255], in RGB format. Can be
            of type float or int. Should be in NHWC format.
    """

    def __init__(self, value_range, bins=256, **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        self.value_range = value_range

    def equalize_channel(self, images, channel_index):
        """equalize_channel performs histogram equalization on a single channel.

        Args:
            image: int Tensor with pixels in range [0, 255], RGB format,
                with channels last
            channel_index: channel to equalize
        """
        is_single_image = tf.rank(images) == 4 and tf.shape(images)[0] == 1

        images = images[..., channel_index]
        # Compute the histogram of the image channel.

        # If the input is not a batch of images, directly using
        # tf.histogram_fixed_width is much faster than using tf.vectorized_map
        if is_single_image:
            histogram = tf.histogram_fixed_width(
                images, [0, 255], nbins=self.bins
            )
            histogram = tf.expand_dims(histogram, axis=0)
        else:
            partial_hist = partial(
                tf.histogram_fixed_width, value_range=[0, 255], nbins=self.bins
            )
            histogram = tf.vectorized_map(
                partial_hist, images, fallback_to_while_loop=True, warn=True
            )

        # For the purposes of computing the step, filter out the non-zeros.
        # Zeroes are replaced by a big number while calculating min to keep
        # shape constant across input sizes for compatibility with
        # vectorized_map

        big_number = 1410065408
        histogram_without_zeroes = tf.where(
            tf.equal(histogram, 0),
            big_number,
            histogram,
        )

        step = (
            tf.reduce_sum(histogram, axis=-1)
            - tf.reduce_min(histogram_without_zeroes, axis=-1)
        ) // (self.bins - 1)

        def build_mapping(histogram, step):
            bacth_size = tf.shape(histogram)[0]

            # Replace where step is 0 with 1 to avoid division by 0.
            # This doesn't change the result, because where step==0 the
            # original image is returned
            _step = tf.where(
                tf.equal(step, 0),
                1,
                step,
            )
            _step = tf.expand_dims(_step, -1)

            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lookup_table = (
                tf.cumsum(histogram, axis=-1) + (_step // 2)
            ) // _step

            # Shift lookup_table, prepending with 0.
            lookup_table = tf.concat(
                [tf.tile([[0]], [bacth_size, 1]), lookup_table[..., :-1]],
                axis=1,
            )

            # Clip the counts to be in range. This is done
            # in the C code for image.point.
            return tf.clip_by_value(lookup_table, 0, 255)

        # If step is zero, return the original image. Otherwise, build
        # lookup table from the full histogram and step and then index from it.
        # The lookup table is built for all images,
        # regardless of the corresponding value of step.
        result = tf.where(
            tf.reshape(tf.equal(step, 0), (-1, 1, 1)),
            images,
            tf.gather(
                build_mapping(histogram, step), images, batch_dims=1, axis=1
            ),
        )

        return result

    def augment_images(self, images, transformations=None, **kwargs):
        images = preprocessing.transform_value_range(
            images, self.value_range, (0, 255), dtype=self.compute_dtype
        )
        images = tf.cast(images, tf.int32)

        images = tf.map_fn(
            lambda channel: self.equalize_channel(images, channel),
            tf.range(tf.shape(images)[-1]),
        )
        images = tf.transpose(images, [1, 2, 3, 0])

        images = tf.cast(images, self.compute_dtype)
        images = preprocessing.transform_value_range(
            images, (0, 255), self.value_range, dtype=self.compute_dtype
        )
        return images

    def augment_bounding_boxes(self, bounding_boxes, **kwargs):
        return bounding_boxes

    def augment_labels(self, labels, transformations=None, **kwargs):
        return labels

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def augment_targets(self, targets, transformations, **kwargs):
        return targets

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"bins": self.bins, "value_range": self.value_range})
        return config
