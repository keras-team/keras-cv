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
"""Common data structures for Waymo Open Dataset inputs."""

import dataclasses
from typing import Optional

import tensorflow as tf


@dataclasses.dataclass
class PointTensors:
    """Wraps point related tensors."""

    # [N, 3] point x, y, z global cartesian coordinates.
    point_xyz: tf.Tensor
    # [N, 4] point feature: intensity, elongation, has_second, is_second.
    point_feature: tf.Tensor
    # [N, 3] range image row, column indices and sensor id.
    point_range_image_row_col_sensor_id: tf.Tensor
    # [N] NLZ (no label zone) mask. Set to true if the point is in NLZ.
    label_point_nlz: tf.Tensor


@dataclasses.dataclass
class LabelTensors:
    """Wraps label related tensors."""

    # [M, 7] 3d boxes in [center_{x,y,z}, length, width, height, heading].
    label_box: Optional[tf.Tensor] = None
    # [M] box id.
    label_box_id: Optional[tf.Tensor] = None
    # [M, 4] box speed_{x,y} and accel_{x,y}.
    label_box_meta: Optional[tf.Tensor] = None
    # [M] box class.
    label_box_class: Optional[tf.Tensor] = None
    # [M] number of points in each box.
    label_box_density: Optional[tf.Tensor] = None
    # [M] detection difficulty level.
    label_box_detection_difficulty: Optional[tf.Tensor] = None
    # [M] valid box mask.
    label_box_mask: Optional[tf.Tensor] = None
    # [M] object class of each point.
    label_point_class: Optional[tf.Tensor] = None
