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

from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Tuple

import tensorflow as tf

from keras_cv import bounding_box


def _feature_bilinear_interpolation(
    features: tf.Tensor, kernel_y: tf.Tensor, kernel_x: tf.Tensor
) -> tf.Tensor:
    """
    Feature bilinear interpolation.

    The RoIAlign feature f can be computed by bilinear interpolation
    of four neighboring feature points f0, f1, f2, and f3.
    f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                          [f10, f11]]
    f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    kernel_y = [hy, ly]
    kernel_x = [hx, lx]

    Args:
      features: The features are in shape of [batch_size, num_boxes, output_size *
        2, output_size * 2, num_filters].
      kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
      kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].

    Returns:
      A 5-D tensor representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """
    features_shape = tf.shape(features)
    batch_size, num_boxes, output_size, num_filters = (
        features_shape[0],
        features_shape[1],
        features_shape[2],
        features_shape[4],
    )

    output_size = output_size // 2
    kernel_y = tf.reshape(kernel_y, [batch_size, num_boxes, output_size * 2, 1])
    kernel_x = tf.reshape(kernel_x, [batch_size, num_boxes, 1, output_size * 2])
    # Use implicit broadcast to generate the interpolation kernel. The
    # multiplier `4` is for avg pooling.
    interpolation_kernel = kernel_y * kernel_x * 4

    # Interpolate the gathered features with computed interpolation kernels.
    features *= tf.cast(
        tf.expand_dims(interpolation_kernel, axis=-1), dtype=features.dtype
    )
    features = tf.reshape(
        features,
        [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters],
    )
    features = tf.nn.avg_pool(features, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
    features = tf.reshape(
        features, [batch_size, num_boxes, output_size, output_size, num_filters]
    )
    return features


def _compute_grid_positions(
    boxes: tf.Tensor, boundaries: tf.Tensor, output_size: int, sample_offset: float
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Computes the grid position w.r.t. the corresponding feature map.

    Args:
      boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
        information of each box w.r.t. the corresponding feature map.
        boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
        corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
          in terms of the number of pixels of the corresponding feature map size.
      boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
        the boundary (in (y, x)) of the corresponding feature map for each box.
        Any resampled grid points that go beyond the bounary will be clipped.
      output_size: a scalar indicating the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample offset
        from grid point.

    Returns:
      kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
      kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].
      box_grid_y0y1: Tensor of size [batch_size, boxes, output_size, 2]
      box_grid_x0x1: Tensor of size [batch_size, boxes, output_size, 2]
    """
    boxes_shape = tf.shape(boxes)
    batch_size, num_boxes = boxes_shape[0], boxes_shape[1]
    if batch_size is None:
        batch_size = tf.shape(boxes)[0]
    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
        box_grid_x.append(
            boxes[:, :, 1] + (i + sample_offset) * boxes[:, :, 3] / output_size
        )
        box_grid_y.append(
            boxes[:, :, 0] + (i + sample_offset) * boxes[:, :, 2] / output_size
        )
    box_grid_x = tf.stack(box_grid_x, axis=2)
    box_grid_y = tf.stack(box_grid_y, axis=2)

    box_grid_y0 = tf.floor(box_grid_y)
    box_grid_x0 = tf.floor(box_grid_x)
    box_grid_x0 = tf.maximum(tf.cast(0.0, dtype=box_grid_x0.dtype), box_grid_x0)
    box_grid_y0 = tf.maximum(tf.cast(0.0, dtype=box_grid_y0.dtype), box_grid_y0)

    box_grid_x0 = tf.minimum(box_grid_x0, tf.expand_dims(boundaries[:, :, 1], -1))
    box_grid_x1 = tf.minimum(box_grid_x0 + 1, tf.expand_dims(boundaries[:, :, 1], -1))
    box_grid_y0 = tf.minimum(box_grid_y0, tf.expand_dims(boundaries[:, :, 0], -1))
    box_grid_y1 = tf.minimum(box_grid_y0 + 1, tf.expand_dims(boundaries[:, :, 0], -1))

    box_gridx0x1 = tf.stack([box_grid_x0, box_grid_x1], axis=-1)
    box_gridy0y1 = tf.stack([box_grid_y0, box_grid_y1], axis=-1)

    # The RoIAlign feature f can be computed by bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3.
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
    # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
    ly = box_grid_y - box_grid_y0
    lx = box_grid_x - box_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx
    kernel_y = tf.reshape(
        tf.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size, 2, 1]
    )
    kernel_x = tf.reshape(
        tf.stack([hx, lx], axis=3), [batch_size, num_boxes, output_size, 2, 1]
    )
    return kernel_y, kernel_x, box_gridy0y1, box_gridx0x1


def multilevel_crop_and_resize(
    features: Dict[str, tf.Tensor],
    boxes: tf.Tensor,
    output_size: int = 7,
    sample_offset: float = 0.5,
) -> tf.Tensor:
    """
    Crop and resize on multilevel feature pyramid.

    Generate the (output_size, output_size) set of pixels for each input box
    by first locating the box into the correct feature level, and then cropping
    and resizing it using the correspoding feature map of that level.

    Args:
      features: A dictionary with key as pyramid level and value as features. The
        features are in shape of [batch_size, height_l, width_l, num_filters].
      boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row represents
        a box with [y1, x1, y2, x2] in un-normalized coordinates.
      output_size: A scalar to indicate the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample offset
        from grid point.

    Returns:
      A 5-D tensor representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """

    with tf.name_scope("multilevel_crop_and_resize"):
        levels = list(features.keys())
        min_level = int(min(levels))
        max_level = int(max(levels))
        features_shape = tf.shape(features[min_level])
        batch_size, max_feature_height, max_feature_width, num_filters = (
            features_shape[0],
            features_shape[1],
            features_shape[2],
            features_shape[3],
        )

        num_boxes = tf.shape(boxes)[1]

        # Stack feature pyramid into a features_all of shape
        # [batch_size, levels, height, width, num_filters].
        features_all = []
        feature_heights = []
        feature_widths = []
        for level in range(min_level, max_level + 1):
            shape = features[level].get_shape().as_list()
            feature_heights.append(shape[1])
            feature_widths.append(shape[2])
            # Concat tensor of [batch_size, height_l * width_l, num_filters] for each
            # levels.
            features_all.append(
                tf.reshape(features[level], [batch_size, -1, num_filters])
            )
        features_r2 = tf.reshape(tf.concat(features_all, 1), [-1, num_filters])

        # Calculate height_l * width_l for each level.
        level_dim_sizes = [
            feature_widths[i] * feature_heights[i] for i in range(len(feature_widths))
        ]
        # level_dim_offsets is accumulated sum of level_dim_size.
        level_dim_offsets = [0]
        for i in range(len(feature_widths) - 1):
            level_dim_offsets.append(level_dim_offsets[i] + level_dim_sizes[i])
        batch_dim_size = level_dim_offsets[-1] + level_dim_sizes[-1]
        level_dim_offsets = tf.constant(level_dim_offsets, tf.int32)
        height_dim_sizes = tf.constant(feature_widths, tf.int32)

        # Assigns boxes to the right level.
        box_width = boxes[:, :, 3] - boxes[:, :, 1]
        box_height = boxes[:, :, 2] - boxes[:, :, 0]
        areas_sqrt = tf.sqrt(
            tf.cast(box_height, tf.float32) * tf.cast(box_width, tf.float32)
        )

        # following the FPN paper to divide by 224.
        levels = tf.cast(
            tf.math.floordiv(
                tf.math.log(tf.math.divide_no_nan(areas_sqrt, 224.0)), tf.math.log(2.0)
            )
            + 4.0,
            dtype=tf.int32,
        )
        # Maps levels between [min_level, max_level].
        levels = tf.minimum(max_level, tf.maximum(levels, min_level))

        # Projects box location and sizes to corresponding feature levels.
        scale_to_level = tf.cast(
            tf.pow(tf.constant(2.0), tf.cast(levels, tf.float32)), dtype=boxes.dtype
        )
        boxes /= tf.expand_dims(scale_to_level, axis=2)
        box_width /= scale_to_level
        box_height /= scale_to_level
        boxes = tf.concat(
            [
                boxes[:, :, 0:2],
                tf.expand_dims(box_height, -1),
                tf.expand_dims(box_width, -1),
            ],
            axis=-1,
        )

        # Maps levels to [0, max_level-min_level].
        levels -= min_level
        level_strides = tf.pow([[2.0]], tf.cast(levels, tf.float32))
        boundary = tf.cast(
            tf.concat(
                [
                    tf.expand_dims(
                        [[tf.cast(max_feature_height, tf.float32)]] / level_strides - 1,
                        axis=-1,
                    ),
                    tf.expand_dims(
                        [[tf.cast(max_feature_width, tf.float32)]] / level_strides - 1,
                        axis=-1,
                    ),
                ],
                axis=-1,
            ),
            boxes.dtype,
        )

        # Compute grid positions.
        kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = _compute_grid_positions(
            boxes, boundary, output_size, sample_offset
        )

        x_indices = tf.cast(
            tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2]),
            dtype=tf.int32,
        )
        y_indices = tf.cast(
            tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2]),
            dtype=tf.int32,
        )

        batch_size_offset = tf.tile(
            tf.reshape(tf.range(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]),
            [1, num_boxes, output_size * 2, output_size * 2],
        )
        # Get level offset for each box. Each box belongs to one level.
        levels_offset = tf.tile(
            tf.reshape(
                tf.gather(level_dim_offsets, levels), [batch_size, num_boxes, 1, 1]
            ),
            [1, 1, output_size * 2, output_size * 2],
        )
        y_indices_offset = tf.tile(
            tf.reshape(
                y_indices * tf.expand_dims(tf.gather(height_dim_sizes, levels), -1),
                [batch_size, num_boxes, output_size * 2, 1],
            ),
            [1, 1, 1, output_size * 2],
        )
        x_indices_offset = tf.tile(
            tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
            [1, 1, output_size * 2, 1],
        )
        indices = tf.reshape(
            batch_size_offset + levels_offset + y_indices_offset + x_indices_offset,
            [-1],
        )

        # TODO(tanzhenyu): replace tf.gather with tf.gather_nd and try to get similar
        # performance.
        features_per_box = tf.reshape(
            tf.gather(features_r2, indices),
            [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters],
        )

        # Bilinear interpolation.
        features_per_box = _feature_bilinear_interpolation(
            features_per_box, kernel_y, kernel_x
        )
        return features_per_box


# TODO(tanzhenyu): Remove this implementation once roi_pool has better performance.
# as this is mostly a duplicate of
# https://github.com/tensorflow/models/blob/master/official/legacy/detection/ops/spatial_transform_ops.py#L324
@tf.keras.utils.register_keras_serializable(package="keras_cv")
class _ROIAligner(tf.keras.layers.Layer):
    """Performs ROIAlign for the second stage processing."""

    def __init__(
        self, bounding_box_format, target_size=7, sample_offset: float = 0.5, **kwargs
    ):
        """
        Generates ROI Aligner.

        Args:
          bounding_box_format: the input format for boxes.
          crop_size: An `int` of the output size of the cropped features.
          sample_offset: A `float` in [0, 1] of the subpixel sample offset.
          **kwargs: Additional keyword arguments passed to Layer.
        """
        self._config_dict = {
            "bounding_box_format": bounding_box_format,
            "crop_size": target_size,
            "sample_offset": sample_offset,
        }
        super().__init__(**kwargs)

    def call(
        self,
        features: Mapping[str, tf.Tensor],
        boxes: tf.Tensor,
        training: Optional[bool] = None,
    ):
        """

        Args:
          features: A dictionary with key as pyramid level and value as features.
            The features are in shape of
            [batch_size, height_l, width_l, num_filters].
          boxes: A 3-D `tf.Tensor` of shape [batch_size, num_boxes, 4]. Each row
            represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
            from grid point.
          training: A `bool` of whether it is in training mode.
        Returns:
          A 5-D `tf.Tensor` representing feature crop of shape
          [batch_size, num_boxes, crop_size, crop_size, num_filters].
        """
        boxes = bounding_box.convert_format(
            boxes, source=self._config_dict["bounding_box_format"], target="yxyx"
        )
        roi_features = multilevel_crop_and_resize(
            features,
            boxes,
            output_size=self._config_dict["crop_size"],
            sample_offset=self._config_dict["sample_offset"],
        )
        return roi_features

    def get_config(self):
        return self._config_dict
