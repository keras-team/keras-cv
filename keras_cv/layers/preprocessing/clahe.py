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
class CLAHE(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """Performs Contrast Limited Adaptive Histogram Equalization on an image.

    Args:
        value_range: A tuple or a list of two elements. The first value represents
            the lower bound for values in passed image, the second represents the
            upper bound. Images passed to the layer should have values within
            `value_range`.
        clip_limit: A floating point value or Tensor.
            0 will result in no clipping (AHE only)
            Limits the noise amplification in near-constant regions.
            Default 4.0
        tile_grid_size: A tensor of shape
            `(tiles_in_x_direction, tiles_in_y_direction)`
            Specifies how many tiles to break the image into.
            Default (8x8).
    Returns:
        Contrast-limited, Adaptive-Histogram equalized image

    Usage:
    ```python
    clahe = CLAHE()

    (images, labels), _ = tf.keras.datasets.cifar10.load_data()
    images = clahe(images)

    Call arguments:
        images: Tensor of pixels in range [0, 255], in RGB format.  Can be
            of type float or int.  Should be in NHWC format.
    """

    def __init__(self, value_range, clip_limit=4.0, tile_grid_size=(8, 8), **kwargs):
        super().__init__(**kwargs)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.value_range = value_range

    def _clahe(self, image):
        original_2d_shape = (tf.shape(image)[0], tf.shape(image)[1])
        original_dtype = image.dtype

        # Need image in int32 format for later gather_nd ops
        image = tf.cast(image, tf.int32)

        tile_shape = tf.truediv(original_2d_shape, self.tile_grid_size)
        tile_shape = tf.cast(tf.math.ceil(tile_shape), tf.int32)

        # Reflection-pad image
        # check if image can be covered by tile
        pad_y = tf.cond(
            original_2d_shape[0] % tile_shape[0] != 0,
            true_fn=lambda: tile_shape[0] - (original_2d_shape[0] % tile_shape[0]),
            false_fn=lambda: 0,
        )
        pad_x = tf.cond(
            original_2d_shape[1] % tile_shape[1] != 0,
            true_fn=lambda: tile_shape[1] - (original_2d_shape[1] % tile_shape[1]),
            false_fn=lambda: 0,
        )

        image_padded = tf.pad(image, [[0, pad_y], [0, pad_x], [0, 0]], "reflect")

        all_tiles = tf.space_to_batch(
            input=tf.expand_dims(image_padded, axis=0),
            block_shape=tile_shape,
            paddings=[[0, 0], [0, 0]],
        )

        # Compute per-tile histogram
        hists = tf.math.reduce_sum(
            tf.one_hot(all_tiles, depth=256, on_value=1, off_value=0, axis=0), axis=1
        )

        # Clip histograms, if necessary
        if self.clip_limit > 0:
            clip_limit_actual = tf.cast(
                self.clip_limit * ((tile_shape[0] * tile_shape[1]) / 256), tf.int32
            )
            clipped_hists = tf.clip_by_value(
                hists, clip_value_min=0, clip_value_max=clip_limit_actual
            )
            # It is advantageous not to discard the part of the histogram that exceeds
            # the clip limit but to redistribute it equally among all histogram bins.
            clipped_px_count = tf.math.reduce_sum(hists - clipped_hists, axis=0)
            clipped_hists = tf.cast(clipped_hists, tf.float32)
            clipped_px_count = tf.cast(clipped_px_count, tf.float32)
            clipped_hists = clipped_hists + tf.math.truediv(clipped_px_count, 256)
        else:
            clipped_hists = tf.cast(hists, tf.float32)

        cdf = tf.math.cumsum(clipped_hists, axis=0)
        cdf_min = tf.math.reduce_min(clipped_hists, axis=0)

        numerator = cdf - cdf_min
        denominator = tf.cast(tile_shape[0] * tile_shape[1], tf.float32) - cdf_min

        cdf_normalized = tf.round(tf.math.divide_no_nan(numerator, denominator) * 255)
        cdf_normalized = tf.cast(cdf_normalized, tf.int32)

        # Reflection-pad the cdf functions so that we don't have to explicitly deal
        # with corners/edge
        cdf_padded = tf.pad(
            cdf_normalized, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC"
        )

        coords = tf.stack(
            tf.meshgrid(
                tf.range(tf.shape(image_padded)[0]),
                tf.range(tf.shape(image_padded)[1]),
                tf.range(tf.shape(image_padded)[2]),
                indexing="ij",
            )
        )

        y_coords = coords[0, :, :]
        x_coords = coords[1, :, :]
        z_coords = coords[2, :, :]

        # Interpolation for efficient computation

        half_tile_shape = tf.math.floordiv(tile_shape, 2)

        nw_y_component = tf.math.floordiv(y_coords - half_tile_shape[0], tile_shape[0])
        nw_x_component = tf.math.floordiv(x_coords - half_tile_shape[1], tile_shape[1])

        # Need to correct negative values because negative-indexing for gather_nd ops
        # not supported on all processors (cdf is padded to account for this)
        nw_y_component = nw_y_component + 1
        nw_x_component = nw_x_component + 1

        ne_y_component = nw_y_component
        ne_x_component = nw_x_component + 1

        sw_y_component = nw_y_component + 1
        sw_x_component = nw_x_component

        se_y_component = sw_y_component
        se_x_component = sw_x_component + 1

        def cdf_transform(x_comp, y_comp):
            gatherable = tf.stack([image_padded, y_comp, x_comp, z_coords], axis=-1)
            return tf.cast(tf.gather_nd(cdf_padded, gatherable), tf.float32)

        nw_transformed = cdf_transform(nw_x_component, nw_y_component)
        ne_transformed = cdf_transform(ne_x_component, ne_y_component)
        sw_transformed = cdf_transform(sw_x_component, sw_y_component)
        se_transformed = cdf_transform(se_x_component, se_y_component)

        y = (y_coords - half_tile_shape[0]) % tile_shape[0]
        y = tf.cast(tf.math.truediv(y, tile_shape[0]), tf.float32)
        x = (x_coords - half_tile_shape[1]) % tile_shape[1]
        x = tf.cast(tf.math.truediv(x, tile_shape[1]), tf.float32)

        # Interpolate
        interpolated = (y * (x * se_transformed + (1 - x) * sw_transformed)) + (
            1 - y
        ) * (x * ne_transformed + (1 - x) * nw_transformed)

        # Return image to original size and dtype
        interpolated = interpolated[
            0 : original_2d_shape[0], 0 : original_2d_shape[1], :
        ]
        interpolated = tf.cast(tf.round(interpolated), original_dtype)

        return interpolated

    def augment_image(self, image, transformation=None):
        image = preprocessing.transform_value_range(
            image, self.value_range, (0, 255), dtype=image.dtype
        )
        image = self._clahe(image)

        image = preprocessing.transform_value_range(image, (0, 255), self.value_range)
        return image

    def augment_label(self, label, transformation=None):
        return label

    def get_config(self):
        config = {
            "clip_rate": self.clip_rate,
            "tile_grid_size": self.tile_grid_size,
            "value_range": self.value_range,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
