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

from typing import Sequence
from typing import Tuple

import numpy as np
import tensorflow as tf

from keras_cv.layers.object_detection3d import voxel_utils

EPSILON = 1e-4
VOXEL_FEATURE_MIN = -1000


def compute_point_voxel_id(
    point_voxel_xyz: tf.Tensor, voxel_spatial_size: Sequence[int]
) -> tf.Tensor:
    """Computes point voxel IDs.

    Args:
      point_voxel_xyz: [B, N, dim] voxel coordinates for each point
      voxel_spatial_size: voxel spatial size

    Returns:
      point_voxel_id: [B, N] unique ID of each voxel.
    """
    batch_size, _, dim = point_voxel_xyz.shape.as_list()
    if batch_size is None:
        batch_size = tf.shape(point_voxel_xyz)[0]
    assert dim == len(voxel_spatial_size), f"{point_voxel_xyz.shape}"

    voxel_spatial_size_prod = [
        np.prod(voxel_spatial_size[i:]).item() for i in range(dim)
    ]
    voxel_spatial_size_prod_shift = voxel_spatial_size_prod[1:] + [1]
    point_voxel_xyz_multiplied = point_voxel_xyz * tf.constant(
        voxel_spatial_size_prod_shift, dtype=point_voxel_xyz.dtype
    )
    # [B, N]
    point_voxel_id = tf.math.reduce_sum(point_voxel_xyz_multiplied, axis=-1)

    if batch_size == 1:
        return point_voxel_id

    batch_multiplier = tf.range(batch_size, dtype=tf.int32) * voxel_spatial_size_prod[0]
    batch_multiplier = batch_multiplier[:, tf.newaxis]
    return point_voxel_id + batch_multiplier


class PointToVoxel(tf.keras.layers.Layer):
    """Voxelization layer."""

    def __init__(
        self,
        voxel_size: Sequence[float],
        spatial_size: Sequence[float],
        **kwargs,
    ):
        """Voxelization layer constructor.

        Args:
          voxel_size: voxel size in each xyz dimension.
          spatial_size: max/min range in each dim in global coordinate frame.
          name: layer name
          **kwargs: additional key value args (e.g. dtype) passed to the parent
            class.
        """
        super().__init__(**kwargs)
        dim = len(voxel_size)
        assert len(spatial_size) == 2 * dim, f"{spatial_size}"

        self._voxel_size = voxel_size
        self._spatial_size = spatial_size

        self._voxel_spatial_size = voxel_utils.compute_voxel_spatial_size(
            spatial_size, self._voxel_size
        )

    # TODO(tanzhenyu): consider using keras masking.
    def call(
        self,
        point_xyz: tf.Tensor,
        point_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Dynamically voxelizes points.

        B: batch_size.
        N: number of points.
        dim: the input dimension.

        Args:
          point_xyz: [B, N, dim] point xyz in global coordinate relative to sdc.
          point_mask: [B, N] valid point mask.

        Returns:
          point_voxel_feature: [B, N, dim] voxel feature (delta_{x,y,z}).
          point_voxel_id: [B, N] voxel ID of each point. Invalid voxels have Id's
            set to 0.
          point_voxel_mask: [B, N] validpoint voxel boolean mask.
        """
        # [B, N, dim]
        # convert from point coordinate to voxel index
        point_voxel_xyz_float = voxel_utils.point_to_voxel_coord(
            point_xyz, self._voxel_size, dtype=point_xyz.dtype
        )
        # [B, N, dim]
        # delta to the nearest voxel
        point_voxel_feature = point_xyz - voxel_utils.voxel_coord_to_point(
            point_voxel_xyz_float, self._voxel_size, dtype=point_xyz.dtype
        )

        # [B, N, dim]
        point_voxel_xyz_int = tf.cast(point_voxel_xyz_float, dtype=tf.int32)
        # [dim]
        # get xmin, ymin, zmin
        voxel_origin = voxel_utils.compute_voxel_origin(
            self._spatial_size, self._voxel_size
        )

        # [B, N, dim]
        # convert point voxel to positive voxel index
        point_voxel_xyz = point_voxel_xyz_int - voxel_origin[tf.newaxis, tf.newaxis, :]

        # [B, N]
        # remove points outside of the voxel boundary
        point_voxel_mask = tf.logical_and(
            point_voxel_xyz >= 0,
            point_voxel_xyz
            < tf.constant(self._voxel_spatial_size, dtype=point_voxel_xyz.dtype),
        )
        point_voxel_mask = tf.math.reduce_all(point_voxel_mask, axis=-1)
        point_voxel_mask = tf.logical_and(point_voxel_mask, point_mask)

        # [B, N]
        point_voxel_mask_int = tf.cast(point_voxel_mask, dtype=tf.int32)
        # [B, N] for voxel_id, int constant for num_voxels, in the range of [0, B * num_voxels]
        point_voxel_id = compute_point_voxel_id(
            point_voxel_xyz, self._voxel_spatial_size
        )
        # [B, N]
        point_voxel_id = point_voxel_id * point_voxel_mask_int

        return point_voxel_feature, point_voxel_id, point_voxel_mask


class DynamicVoxelization(tf.keras.layers.Layer):
    """Dynamic voxelization and pool layer.

    This layer assigns and pools points into voxels,
    then it concatenates with point features and feed into a neural network,
    and max pools all point features inside each voxel.

    Args:
      point_net: a keras Layer that project point feature into another dimension.
      voxel_size: the x, y, z dimension of each voxel.
      spatial_size: the x, y, z boundary of voxels

    Returns:
      voxelized feature, a float Tensor.

    """

    def __init__(
        self,
        point_net: tf.keras.layers.Layer,
        voxel_size: Sequence[float],
        spatial_size: Sequence[float],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._point_net = point_net
        self._voxelization_layer = PointToVoxel(
            voxel_size=voxel_size, spatial_size=spatial_size
        )
        self._voxel_size = voxel_size
        self._spatial_size = spatial_size
        self._voxel_spatial_size = voxel_utils.compute_voxel_spatial_size(
            spatial_size, self._voxel_size
        )
        self._voxel_spatial_size_volume = np.prod(self._voxel_spatial_size).item()

    def call(
        self,
        point_xyz: tf.Tensor,
        point_feature: tf.Tensor,
        point_mask: tf.Tensor,
        training: bool,
    ) -> tf.Tensor:
        """Voxelizes and learns voxel features with a point net.

        B: batch_size.
        N: number of points.
        dim: the input dimension.

        Args:
          point_xyz: [B, N, 3] point xyz in global coordinate.
          point_feature: [B, N, dim] point feature inputs.
          point_mask: [B, N] valid point mask.
          training: whether it is in training mode.

        Returns:
          voxel_feature: [B, x_max, y_max, {z_max,}, mlp_dimension] voxel
            features. If z_max is 1, z-dim is squeezed.
        """
        (
            point_voxel_feature,
            point_voxel_id,
            point_voxel_mask,
        ) = self._voxelization_layer(point_xyz=point_xyz, point_mask=point_mask)
        # TODO(tanzhenyu): move compute_point_voxel_id to here, so PointToVoxel layer is more generic.
        point_feature = tf.concat([point_feature, point_voxel_feature], axis=-1)
        batch_size = point_feature.shape.as_list()[0] or tf.shape(point_feature)[0]
        # [B, N, 1]
        point_mask_float = tf.cast(point_voxel_mask, point_feature.dtype)[
            ..., tf.newaxis
        ]
        # [B, N, dim]
        point_feature = point_feature * point_mask_float
        point_feature = self._point_net(
            point_feature, mask=point_mask, training=training
        )
        # [B, N, new_dim]
        point_feature = point_feature * point_mask_float
        new_dim = point_feature.shape.as_list()[-1]
        point_feature = tf.reshape(point_feature, [-1, new_dim])
        point_voxel_id = tf.reshape(point_voxel_id, [-1])
        # [B * num_voxels, new_dim]
        voxel_feature = tf.math.unsorted_segment_max(
            point_feature, point_voxel_id, batch_size * self._voxel_spatial_size_volume
        )
        # unsorted_segment_max sets empty values to -inf(float).
        voxel_feature_valid_mask = voxel_feature > VOXEL_FEATURE_MIN
        voxel_feature = voxel_feature * tf.cast(
            voxel_feature_valid_mask, dtype=voxel_feature.dtype
        )
        out_shape = [batch_size] + self._voxel_spatial_size + [new_dim]
        if out_shape[-2] == 1:
            out_shape = out_shape[:-2] + [out_shape[-1]]
        voxel_feature = tf.reshape(voxel_feature, out_shape)
        return voxel_feature
