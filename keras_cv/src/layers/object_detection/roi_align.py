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

from typing import Optional

from keras_cv.src import bounding_box
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


def _feature_bilinear_interpolation(features, kernel_y, kernel_x):
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
      features: The features are in shape of [batch_size, num_boxes,
        output_size * 2, output_size * 2, num_filters].
      kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
      kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].

    Returns:
      A 5-D tensor representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """
    features_shape = ops.shape(features)
    batch_size, num_boxes, output_size, num_filters = (
        features_shape[0],
        features_shape[1],
        features_shape[2],
        features_shape[4],
    )

    output_size = output_size // 2
    kernel_y = ops.reshape(
        kernel_y, [batch_size, num_boxes, output_size * 2, 1]
    )
    kernel_x = ops.reshape(
        kernel_x, [batch_size, num_boxes, 1, output_size * 2]
    )
    # Use implicit broadcast to generate the interpolation kernel. The
    # multiplier `4` is for avg pooling.
    interpolation_kernel = kernel_y * kernel_x * 4

    # Interpolate the gathered features with computed interpolation kernels.
    features *= ops.cast(
        ops.expand_dims(interpolation_kernel, axis=-1), dtype=features.dtype
    )
    features = ops.reshape(
        features,
        [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters],
    )
    features = ops.nn.average_pool(
        features, [1, 2, 2, 1], [1, 2, 2, 1], "VALID"
    )
    features = ops.reshape(
        features, [batch_size, num_boxes, output_size, output_size, num_filters]
    )
    return features


def _compute_grid_positions(
    boxes,
    boundaries,
    output_size,
    sample_offset,
):
    """
    Computes the grid position w.r.t. the corresponding feature map.

    Args:
      boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
        information of each box w.r.t. the corresponding feature map.
        boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
        corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
          in terms of the number of pixels of the corresponding feature map
          size.
      boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
        the boundary (in (y, x)) of the corresponding feature map for each box.
        Any resampled grid points that go beyond the boundary will be clipped.
      output_size: a scalar indicating the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample
        offset from grid point.

    Returns:
      kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
      kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].
      box_grid_y0y1: Tensor of size [batch_size, boxes, output_size, 2]
      box_grid_x0x1: Tensor of size [batch_size, boxes, output_size, 2]
    """
    boxes_shape = ops.shape(boxes)
    batch_size, num_boxes = boxes_shape[0], boxes_shape[1]
    if batch_size is None:
        batch_size = ops.shape(boxes)[0]
    box_grid_x = []
    box_grid_y = []
    for i in range(output_size):
        box_grid_x.append(
            boxes[:, :, 1] + (i + sample_offset) * boxes[:, :, 3] / output_size
        )
        box_grid_y.append(
            boxes[:, :, 0] + (i + sample_offset) * boxes[:, :, 2] / output_size
        )
    box_grid_x = ops.stack(box_grid_x, axis=2)
    box_grid_y = ops.stack(box_grid_y, axis=2)

    box_grid_y0 = ops.floor(box_grid_y)
    box_grid_x0 = ops.floor(box_grid_x)
    box_grid_x0 = ops.maximum(
        ops.cast(0.0, dtype=box_grid_x0.dtype), box_grid_x0
    )
    box_grid_y0 = ops.maximum(
        ops.cast(0.0, dtype=box_grid_y0.dtype), box_grid_y0
    )

    box_grid_x0 = ops.minimum(
        box_grid_x0, ops.expand_dims(boundaries[:, :, 1], -1)
    )
    box_grid_x1 = ops.minimum(
        box_grid_x0 + 1, ops.expand_dims(boundaries[:, :, 1], -1)
    )
    box_grid_y0 = ops.minimum(
        box_grid_y0, ops.expand_dims(boundaries[:, :, 0], -1)
    )
    box_grid_y1 = ops.minimum(
        box_grid_y0 + 1, ops.expand_dims(boundaries[:, :, 0], -1)
    )

    box_gridx0x1 = ops.stack([box_grid_x0, box_grid_x1], axis=-1)
    box_gridy0y1 = ops.stack([box_grid_y0, box_grid_y1], axis=-1)

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
    kernel_y = ops.reshape(
        ops.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size, 2, 1]
    )
    kernel_x = ops.reshape(
        ops.stack([hx, lx], axis=3), [batch_size, num_boxes, output_size, 2, 1]
    )
    return kernel_y, kernel_x, box_gridy0y1, box_gridx0x1


def multilevel_crop_and_resize(
    features,
    boxes,
    output_size: int = 7,
    sample_offset: float = 0.5,
):
    """
    Crop and resize on multilevel feature pyramid.

    Generate the (output_size, output_size) set of pixels for each input box
    by first locating the box into the correct feature level, and then cropping
    and resizing it using the corresponding feature map of that level.

    Args:
      features: A dictionary with key as pyramid level and value as features.
        The pyramid level keys need to be represented by strings like so:
        "P2", "P3", "P4", and so on.
        The features are in shape of [batch_size, height_l, width_l,
        num_filters].
      boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row
        represents a box with [y1, x1, y2, x2] in un-normalized coordinates.
      output_size: A scalar to indicate the output crop size.
      sample_offset: a float number in [0, 1] indicates the subpixel sample
        offset from grid point.

    Returns:
      A 5-D tensor representing feature crop of shape
      [batch_size, num_boxes, output_size, output_size, num_filters].
    """

    levels_str = list(features.keys())
    # Levels are represented by strings with a prefix "P" to represent
    # pyramid levels. The integer level can be obtained by looking at
    # the value that follows the "P".
    levels = [int(level_str[1:]) for level_str in levels_str]
    min_level = min(levels)
    max_level = max(levels)
    features_shape = ops.shape(features[f"P{min_level}"])
    batch_size, max_feature_height, max_feature_width, num_filters = (
        features_shape[0],
        features_shape[1],
        features_shape[2],
        features_shape[3],
    )

    num_boxes = ops.shape(boxes)[1]

    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    feature_heights = []
    feature_widths = []
    for level in range(min_level, max_level + 1):
        shape = ops.shape(features[f"P{level}"])
        feature_heights.append(shape[1])
        feature_widths.append(shape[2])
        # Concat tensor of [batch_size, height_l * width_l, num_filters] for
        # each level.
        features_all.append(
            ops.reshape(features[f"P{level}"], [batch_size, -1, num_filters])
        )
    features_r2 = ops.reshape(
        ops.concatenate(features_all, 1), [-1, num_filters]
    )

    # Calculate height_l * width_l for each level.
    level_dim_sizes = [
        feature_widths[i] * feature_heights[i]
        for i in range(len(feature_widths))
    ]
    # level_dim_offsets is accumulated sum of level_dim_size.
    level_dim_offsets = [0]
    for i in range(len(feature_widths) - 1):
        level_dim_offsets.append(level_dim_offsets[i] + level_dim_sizes[i])
    batch_dim_size = level_dim_offsets[-1] + level_dim_sizes[-1]
    level_dim_offsets = (
        ops.ones_like(level_dim_offsets, dtype="int32") * level_dim_offsets
    )
    height_dim_sizes = (
        ops.ones_like(feature_widths, dtype="int32") * feature_widths
    )

    # Assigns boxes to the right level.
    box_width = boxes[:, :, 3] - boxes[:, :, 1]
    box_height = boxes[:, :, 2] - boxes[:, :, 0]
    areas_sqrt = ops.sqrt(
        ops.cast(box_height, "float32") * ops.cast(box_width, "float32")
    )

    # following the FPN paper to divide by 224.
    levels = ops.cast(
        ops.floor_divide(
            ops.log(ops.divide(areas_sqrt, 224.0)),
            ops.log(2.0),
        )
        + 4.0,
        dtype="int32",
    )
    # Maps levels between [min_level, max_level].
    levels = ops.minimum(max_level, ops.maximum(levels, min_level))

    # Projects box location and sizes to corresponding feature levels.
    scale_to_level = ops.cast(
        ops.power(2.0, ops.cast(levels, "float32")),
        dtype=boxes.dtype,
    )
    boxes /= ops.expand_dims(scale_to_level, axis=2)
    box_width /= scale_to_level
    box_height /= scale_to_level
    boxes = ops.concatenate(
        [
            boxes[:, :, 0:2],
            ops.expand_dims(box_height, -1),
            ops.expand_dims(box_width, -1),
        ],
        axis=-1,
    )

    # Maps levels to [0, max_level-min_level].
    levels -= min_level
    level_strides = ops.power([[2.0]], ops.cast(levels, "float32"))
    boundary = ops.cast(
        ops.concatenate(
            [
                ops.expand_dims(
                    [[ops.cast(max_feature_height, "float32")]] / level_strides
                    - 1,
                    axis=-1,
                ),
                ops.expand_dims(
                    [[ops.cast(max_feature_width, "float32")]] / level_strides
                    - 1,
                    axis=-1,
                ),
            ],
            axis=-1,
        ),
        boxes.dtype,
    )

    # Compute grid positions.
    (
        kernel_y,
        kernel_x,
        box_gridy0y1,
        box_gridx0x1,
    ) = _compute_grid_positions(boxes, boundary, output_size, sample_offset)

    x_indices = ops.cast(
        ops.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2]),
        dtype="int32",
    )
    y_indices = ops.cast(
        ops.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2]),
        dtype="int32",
    )

    batch_size_offset = ops.tile(
        ops.reshape(
            ops.arange(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]
        ),
        [1, num_boxes, output_size * 2, output_size * 2],
    )
    # Get level offset for each box. Each box belongs to one level.
    levels_offset = ops.tile(
        ops.reshape(
            ops.take(level_dim_offsets, levels),
            [batch_size, num_boxes, 1, 1],
        ),
        [1, 1, output_size * 2, output_size * 2],
    )
    y_indices_offset = ops.tile(
        ops.reshape(
            y_indices * ops.expand_dims(ops.take(height_dim_sizes, levels), -1),
            [batch_size, num_boxes, output_size * 2, 1],
        ),
        [1, 1, 1, output_size * 2],
    )
    x_indices_offset = ops.tile(
        ops.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
        [1, 1, output_size * 2, 1],
    )
    indices = ops.reshape(
        batch_size_offset + levels_offset + y_indices_offset + x_indices_offset,
        [-1],
    )

    # TODO(tanzhenyu): replace tf.gather with tf.gather_nd and try to get
    #  similar performance.
    features_per_box = ops.reshape(
        ops.take(features_r2, indices),
        [
            batch_size,
            num_boxes,
            output_size * 2,
            output_size * 2,
            num_filters,
        ],
    )

    # Bilinear interpolation.
    features_per_box = _feature_bilinear_interpolation(
        features_per_box, kernel_y, kernel_x
    )
    return features_per_box


# TODO(tanzhenyu): Remove this implementation once roi_pool has better
#  performance as this is mostly a duplicate of
#  https://github.com/tensorflow/models/blob/master/official/legacy/detection/ops/spatial_transform_ops.py#L324
@keras.utils.register_keras_serializable(package="keras_cv")
class _ROIAligner(keras.layers.Layer):
    """Performs ROIAlign for the second stage processing."""

    def __init__(
        self,
        bounding_box_format,
        target_size=7,
        sample_offset: float = 0.5,
        **kwargs,
    ):
        """
        Generates ROI Aligner.

        Args:
          bounding_box_format: the input format for boxes.
          crop_size: An `int` of the output size of the cropped features.
          sample_offset: A `float` in [0, 1] of the subpixel sample offset.
          **kwargs: Additional keyword arguments passed to Layer.
        """
        # assert_tf_keras("keras_cv.layers._ROIAligner")
        self._config_dict = {
            "bounding_box_format": bounding_box_format,
            "crop_size": target_size,
            "sample_offset": sample_offset,
        }
        super().__init__(**kwargs)

    def call(
        self,
        features,
        boxes,
        training: Optional[bool] = None,
    ):
        """

        Args:
          features: A dictionary with key as pyramid level and value as
            features. The features are in shape of
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
            boxes,
            source=self._config_dict["bounding_box_format"],
            target="yxyx",
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
