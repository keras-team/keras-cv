# Copyright 2023 The KerasCV Authors
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

# TODO(ianstenbit): Clean up this anchor generation.
# It currently includes lots of unecessary indirection.


def get_feature_sizes(input_shape, pyramid_levels=[3, 7]):
    # https://github.com/google/automl/tree/master/efficientdet/utils.py#L509
    feature_sizes = [input_shape[:2]]
    for _ in range(max(pyramid_levels)):
        pre_feat_size = feature_sizes[-1]
        feature_sizes.append(
            ((pre_feat_size[0] - 1) // 2 + 1, (pre_feat_size[1] - 1) // 2 + 1)
        )  # ceil mode, like padding="SAME" downsampling
    return feature_sizes


def get_anchors(
    image_shape=(512, 512, 3),
    pyramid_levels=[3, 5],
    aspect_ratios=[1],
    num_scales=1,
    anchor_scale=1,
    grid_zero_start=False,
):
    # base anchors
    scales = [2 ** (ii / num_scales) * anchor_scale for ii in range(num_scales)]
    aspect_ratios_tensor = tf.constant(aspect_ratios, dtype="float32")
    if len(aspect_ratios_tensor.shape) == 1:
        # aspect_ratios = [0.5, 1, 2]
        sqrt_ratios = tf.sqrt(aspect_ratios_tensor)
        ww_ratios, hh_ratios = sqrt_ratios, 1 / sqrt_ratios
    else:
        # aspect_ratios = [(1, 1), (1.4, 0.7), (0.7, 1.4)]
        ww_ratios, hh_ratios = (
            aspect_ratios_tensor[:, 0],
            aspect_ratios_tensor[:, 1],
        )
    base_anchors_hh = tf.reshape(
        tf.expand_dims(scales, 1) * tf.expand_dims(hh_ratios, 0), [-1]
    )
    base_anchors_ww = tf.reshape(
        tf.expand_dims(scales, 1) * tf.expand_dims(ww_ratios, 0), [-1]
    )
    base_anchors_hh_half, base_anchors_ww_half = (
        base_anchors_hh / 2,
        base_anchors_ww / 2,
    )
    base_anchors = tf.stack(
        [
            base_anchors_hh_half * -1,
            base_anchors_ww_half * -1,
            base_anchors_hh_half,
            base_anchors_ww_half,
        ],
        axis=1,
    )

    # make grid
    pyramid_levels = list(range(min(pyramid_levels), max(pyramid_levels) + 1))
    feature_sizes = get_feature_sizes(image_shape, pyramid_levels)

    all_anchors = []
    for level in pyramid_levels:
        stride_hh, stride_ww = (
            feature_sizes[0][0] / feature_sizes[level][0],
            feature_sizes[0][1] / feature_sizes[level][1],
        )
        top, left = (
            (0, 0) if grid_zero_start else (stride_hh / 2, stride_ww / 2)
        )
        hh_centers = tf.range(top, image_shape[0], stride_hh)
        ww_centers = tf.range(left, image_shape[1], stride_ww)
        ww_grid, hh_grid = tf.meshgrid(ww_centers, hh_centers)
        grid = tf.reshape(
            tf.stack([hh_grid, ww_grid, hh_grid, ww_grid], 2), [-1, 1, 4]
        )
        anchors = tf.expand_dims(
            base_anchors * [stride_hh, stride_ww, stride_hh, stride_ww], 0
        ) + tf.cast(grid, base_anchors.dtype)
        anchors = tf.reshape(anchors, [-1, 4])
        all_anchors.append(anchors)
    all_anchors = tf.concat(all_anchors, axis=0) / [
        image_shape[0],
        image_shape[1],
        image_shape[0],
        image_shape[1],
    ]

    return tf.cast(all_anchors, tf.float32)
