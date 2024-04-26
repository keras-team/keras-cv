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

import math
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import ops
from keras_cv.src.backend import scope
from keras_cv.src.backend.scope import tf_data
from keras_cv.src.layers.object_detection_3d import voxel_utils

# Infinite voxel size.
INF_VOXEL_SIZE = 100


def _meshgrid(
    max_radius_in_voxels: Sequence[int], voxel_size: Sequence[float]
) -> np.ndarray:
    """Computes the mesh grid given number of points in each dimension.

    NOTE: this is a pure numpy function.

    Args:
      max_radius_in_voxels: max radius in each dimension in units of voxels.
      voxel_size: voxel size of each dimension.

    Returns:
      point tensor of shape [-1, len(voxel_size)].
    """
    m = max_radius_in_voxels
    dim = len(m)
    assert dim == 2 or dim == 3
    if dim == 2:
        mesh = np.mgrid[-m[0] : m[0] + 1, -m[1] : m[1] + 1]
    else:
        mesh = np.mgrid[-m[0] : m[0] + 1, -m[1] : m[1] + 1, -m[2] : m[2] + 1]
    mesh = np.concatenate(mesh[..., np.newaxis], axis=-1)
    mesh = np.reshape(mesh, [-1, dim])
    return mesh * voxel_size


@tf_data
def compute_heatmap(
    box_3d: tf.Tensor,
    box_mask: tf.Tensor,
    voxel_size: Sequence[float],
    max_radius: Sequence[float],
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compute heatmap for boxes.

    Args:
      box_3d: 3d boxes in xyz format, vehicle frame, [B, boxes, 7].
      box_mask: box masking, [B, boxes]
      voxel_size: the size on each voxel dimension (xyz)
      max_radius: the maximum radius on each voxel dimension (xyz)

    Returns:
      point_xyz: the point location w.r.t. vehicle frame, [B, boxes,
        max_voxels_per_box, 3]
      mask: point mask, [B, boxes, max_voxels_per_box]
      heatmap: the returned heatmap w.r.t box frame, [B, boxes,
        max_voxels_per_box]
      box_id: the box id each point belongs to, [B, boxes, max_voxels_per_box]

    """
    # convert radius from point unit to voxel unit.
    max_radius_in_voxels = [
        math.ceil(mr / vs) for mr, vs in zip(max_radius, voxel_size)
    ]
    # get the mesh grid based on max radius w.r.t each box
    # [max_num_voxels_per_box, 3]
    points_numpy = _meshgrid(max_radius_in_voxels, voxel_size=voxel_size)

    box_center = box_3d[:, :, :3]
    # voxelize and de-voxelize point_xyz
    # This ensures that we are computing heatmap for each voxel with these
    # quantized x,y,z.
    # [B, N, max_num_voxels_per_box, 3]
    point_xyz = (
        box_center[:, :, tf.newaxis, :]
        + tf.constant(points_numpy, dtype=tf.float32)[
            tf.newaxis, tf.newaxis, :, :
        ]
    )
    # [B, N, max_num_voxels_per_box, 3]
    point_xyz = voxel_utils.point_to_voxel_coord(
        point_xyz, voxel_size, dtype=tf.int32
    )
    # Map voxel back to xyz to get quantized version.
    # [B, N, max_num_voxels_per_box, 3]
    point_xyz = voxel_utils.voxel_coord_to_point(
        point_xyz, voxel_size, dtype=tf.float32
    )

    # Transforms these points to the box frame from vehicle frame.
    heading = box_3d[:, :, -1]
    # [B, N, 3, 3]
    rot = voxel_utils.get_yaw_rotation(heading)
    # [B, N, max_num_voxels_per_box, 3]
    point_xyz_rot = tf.linalg.matmul(point_xyz, rot)
    # convert from box frame to vehicle frame.
    # [B, N, max_num_voxels_per_box, 3]
    point_xyz_transform = (
        point_xyz_rot
        + voxel_utils.inv_loc(rot, box_center)[:, :, tf.newaxis, :]
    )
    # Due to the transform above, z=0 can be transformed to a non-zero value.
    # For 2d heatmap, we do not want to use z.
    if voxel_size[2] > INF_VOXEL_SIZE:
        point_xyz_transform = tf.concat(
            [
                point_xyz_transform[..., :2],
                tf.zeros_like(point_xyz_transform[..., :1]),
            ],
            axis=-1,
        )

    # The Gaussian radius is set as the dimension of the boxes
    # [B, N, 3]
    radius = box_3d[:, :, 3:6]
    # [B, N, 1, 3]
    radius = radius[:, :, tf.newaxis, :]
    # The Gaussian standard deviation is set as 1.
    # [B, N, 1, 3]
    sigma = tf.ones_like(radius, dtype=radius.dtype)

    # Compute point mask. Anything outside the radius is invalid.
    # [B, N, max_num_voxels_per_box, 3]
    mask = tf.math.less_equal(tf.math.abs(point_xyz_transform), radius)
    # [B, N, max_num_voxels_per_box]
    mask = tf.math.reduce_all(mask, axis=-1)
    # [B, N, max_num_voxels_per_box]
    mask = tf.logical_and(box_mask[:, :, tf.newaxis], mask)

    # [B, N, max_num_voxels_per_box]
    # Gaussian kernel
    p2 = point_xyz_transform * point_xyz_transform
    p2_sigma = p2 * (-0.5 / (sigma * sigma))
    # in box frame.
    heatmap = tf.exp(tf.reduce_sum(p2_sigma, axis=-1))

    (
        batch_size,
        num_box,
        max_num_voxels_per_box,
        _,
    ) = ops.shape(point_xyz)
    box_id = tf.range(num_box, dtype=tf.int32)
    box_id = tf.tile(
        box_id[tf.newaxis, :, tf.newaxis],
        [batch_size, 1, max_num_voxels_per_box],
    )

    point_xyz = tf.reshape(
        point_xyz, [batch_size, num_box * max_num_voxels_per_box, 3]
    )
    heatmap = tf.reshape(
        heatmap, [batch_size, num_box * max_num_voxels_per_box]
    )
    box_id = tf.reshape(box_id, [batch_size, num_box * max_num_voxels_per_box])
    mask = tf.reshape(mask, [batch_size, num_box * max_num_voxels_per_box])

    return point_xyz, mask, heatmap, box_id


def scatter_to_dense_heatmap(
    point_xyz: tf.Tensor,
    point_mask: tf.Tensor,
    point_box_id: tf.Tensor,
    heatmap: tf.Tensor,
    voxel_size: Sequence[float],
    spatial_size: Sequence[float],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Scatter the heatmap to a dense grid.

    N = num_boxes * max_voxels_per_box

    Args:
      point_xyz: [B, N, 3] 3d points, point coordinate in vehicle frame.
      point_mask: [B, N] valid point mask.
      point_box_id: [B, N] box id of each point. The ID indexes into the input
        box tensors. See compute_heatmap for more details.
      heatmap: [B, N] heatmap value of each point.
      voxel_size: voxel size.
      spatial_size: the spatial size.

    Returns:
      dense_heatmap: [B, H, W] heatmap value.
      dense_box_id: [B, H, W] box id associated with each feature map pixel.
        Only pixels with positive heatmap value have valid box id set. Other
        locations have random values.

    """
    # [B, N, 3]
    # convert to voxel units.
    point_voxel_xyz = voxel_utils.point_to_voxel_coord(
        point_xyz, voxel_size, dtype=tf.int32
    )
    # [3]
    voxel_origin = voxel_utils.compute_voxel_origin(spatial_size, voxel_size)
    # [B, N, 3]
    # shift point voxel coordinates to positive voxel index.
    point_voxel_xyz = point_voxel_xyz - voxel_origin[tf.newaxis, tf.newaxis, :]
    voxel_spatial_size = voxel_utils.compute_voxel_spatial_size(
        spatial_size, voxel_size
    )
    # [B, N]
    point_voxel_valid_mask = tf.math.reduce_all(
        tf.math.logical_and(
            point_voxel_xyz >= 0, point_voxel_xyz < voxel_spatial_size
        ),
        axis=-1,
    )
    # [B, N]
    point_voxel_valid_mask = tf.math.logical_and(
        point_voxel_valid_mask, point_mask
    )
    # [B, N]
    point_voxel_xyz = point_voxel_xyz * tf.cast(
        point_voxel_valid_mask[..., tf.newaxis], dtype=point_voxel_xyz.dtype
    )
    # [B, N]
    # filtered heatmap with out of range voxels.
    heatmap = heatmap * tf.cast(point_voxel_valid_mask, dtype=heatmap.dtype)

    # TODO(tanzheny): consider a batched implementation.
    def fn(args):
        """Calls scatter update."""
        point_voxel_xyz_i, mask_i, heatmap_i, point_box_id_i = args
        mask_index = tf.where(mask_i)

        point_voxel_xyz_i = tf.cast(
            tf.gather_nd(point_voxel_xyz_i, mask_index), tf.int32
        )
        heatmap_i = tf.gather_nd(heatmap_i, mask_index)
        point_box_id_i = tf.gather_nd(point_box_id_i, mask_index)

        # scatter from local heatmap to global heatmap based on point_xyz voxel
        # units
        dense_heatmap_i = tf.tensor_scatter_nd_update(
            tf.zeros(voxel_spatial_size, dtype=heatmap_i.dtype),
            point_voxel_xyz_i,
            heatmap_i,
        )
        dense_box_id_i = tf.tensor_scatter_nd_update(
            tf.zeros(voxel_spatial_size, dtype=tf.int32),
            point_voxel_xyz_i,
            point_box_id_i,
        )
        return dense_heatmap_i, dense_box_id_i

    dense_heatmap, dense_box_id = tf.map_fn(
        fn,
        elems=[point_voxel_xyz, point_voxel_valid_mask, heatmap, point_box_id],
        fn_output_signature=(heatmap.dtype, point_box_id.dtype),
    )

    return dense_heatmap, dense_box_id


def decode_tensor(
    t: tf.Tensor, dims: Sequence[Union[tf.Tensor, int]]
) -> tf.Tensor:
    """

    Args:
      t: int32 or int64 tensor of shape [shape], [B, k]
      dims: list of ints., [H, W, Z]

    Returns:
      t_decoded: int32 or int64 decoded tensor of shape [shape, len(dims)],
        [B, k, 3]
    """
    with tf.name_scope("decode_tensor"):
        multipliers = []
        multiplier = 1
        assert dims
        for d in reversed(dims):
            multipliers.append(multiplier)
            multiplier = multiplier * d
        multipliers = list(reversed(multipliers))

        t_decoded_list = []
        remainder = t
        for m in multipliers:
            t_decoded_list.append(tf.math.floordiv(remainder, m))
            remainder = tf.math.floormod(remainder, m)
        return tf.stack(t_decoded_list, axis=-1)


@tf_data
def compute_top_k_heatmap_idx(heatmap: tf.Tensor, k: int) -> tf.Tensor:
    """Computes the top_k heatmap indices.
    Args:
      heatmap: [B, H, W] for 2 dimension or [B, H, W, Z] for 3 dimensions
      k: integer, represent top_k
    Returns:
      top_k_index: [B, k, 2] for 2 dimensions or [B, k, 3] for 3 dimensions
    """
    shape = ops.shape(heatmap)

    # [B, H*W*Z]
    heatmap_reshape = tf.reshape(heatmap, [shape[0], -1])
    # [B, k]
    # each index in the range of [0, H*W*Z)
    _, indices = tf.math.top_k(heatmap_reshape, k=k, sorted=False)
    # [B, k, 2] or [B, k, 3]
    # shape[1:] = [H, W, Z], convert the indices from 1 dimension to 3
    # dimensions in the range of [0, H), [0, W), [0, Z)
    res = decode_tensor(indices, shape[1:])
    return res


@keras_cv_export("keras_cv.layers.CenterNetLabelEncoder")
class CenterNetLabelEncoder(keras.layers.Layer):
    """Transforms the raw sparse labels into class specific dense training
    labels.

    This layer takes the box locations, box classes and box masks, voxelizes
    and compute the Gaussian radius for each box, then computes class specific
    heatmap for classification and class specific box offset w.r.t to feature
    map for regression.

    Args:
      voxel_size: the x, y, z dimension (in meters) of each voxel.
      max_radius: maximum Gaussian radius in each dimension in meters.
      spatial_size: the x, y, z boundary of voxels
      num_classes: number of object classes.
      top_k_heatmap: A sequence of integers, top k for each class. Can be None.
    """

    def __init__(
        self,
        voxel_size: Sequence[float],
        max_radius: Sequence[float],
        spatial_size: Sequence[float],
        num_classes: int,
        top_k_heatmap: Sequence[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._voxel_size = voxel_size
        self._max_radius = max_radius
        self._spatial_size = spatial_size
        self._num_classes = num_classes
        self._top_k_heatmap = top_k_heatmap

    def call(self, inputs):
        """
        Args:
          inputs: dictionary of Tensors representing a batch of data. Must
          contain 3D box targets under the key "3d_boxes".
        Returns:
          A dictionary of Tensors with all of the original inputs, plus, for
          each class, a new key with encoded CenterNet targets in the format:
          ```
          "class_{class_index}": {
            "heatmap": float Tensor [B, H, W, Z] or [B, H, W]
            "boxes": float Tensor [B, H, W, Z, 7] or [B, H, W, 7]
            "tok_k_index": int Tensor [B, k, 3] or [B, k, 2]
          }
          ```
        where:
          H: number of voxels in y dimension
          W: number of voxels in x dimension
          Z: number of voxels in z dimension
          k: `top_k_heatmap` slice
        """
        with scope.TFDataScope():
            box_3d = inputs["3d_boxes"]["boxes"]
            box_mask = inputs["3d_boxes"]["mask"]
            box_classes = inputs["3d_boxes"]["classes"]
            # point_xyz - [B, num_boxes * max_num_voxels_per_box, 3]
            # heatmap - [B, num_boxes * max_num_voxels_per_box]
            # compute localized heatmap around its radius.
            point_xyz, point_mask, heatmap, box_id = compute_heatmap(
                box_3d,
                box_mask,
                self._voxel_size,
                self._max_radius,
            )
            # heatmap - [B, H, W, Z]
            # scatter the localized heatmap to global heatmap in vehicle frame.
            dense_heatmap, dense_box_id = scatter_to_dense_heatmap(
                point_xyz,
                point_mask,
                box_id,
                heatmap,
                self._voxel_size,
                self._spatial_size,
            )
            b, h, w, z = ops.shape(dense_box_id)
            # [B, H * W * Z]
            dense_box_id = tf.reshape(dense_box_id, [b, h * w * z])
            # mask out invalid boxes to 0, which represents background
            box_classes = box_classes * tf.cast(box_mask, box_classes.dtype)
            # [B, H, W, Z]
            dense_box_classes = tf.reshape(
                tf.gather(box_classes, dense_box_id, batch_dims=1), [b, h, w, z]
            )
            # [B, H, W, Z, 7] in vehicle frame.
            dense_box_3d = tf.reshape(
                tf.gather(box_3d, dense_box_id, batch_dims=1), [b, h, w, z, -1]
            )
            global_xyz = tf.zeros([b, 3], dtype=point_xyz.dtype)
            # [B, H, W, Z, 3]
            feature_map_ref_xyz = voxel_utils.compute_feature_map_ref_xyz(
                self._voxel_size, self._spatial_size, global_xyz
            )
            # convert from global box point xyz to offset w.r.t center of
            # feature map.
            # [B, H, W, Z, 3]
            dense_box_3d_center = dense_box_3d[..., :3] - tf.cast(
                feature_map_ref_xyz, dense_box_3d.dtype
            )
            # [B, H, W, Z, 7]
            dense_box_3d = tf.concat(
                [dense_box_3d_center, dense_box_3d[..., 3:]], axis=-1
            )

            centernet_targets = {}
            for i in range(self._num_classes):
                # Object class is 1-indexed (0 is background).
                dense_box_classes_i = tf.cast(
                    tf.math.equal(dense_box_classes, i + 1),
                    dtype=dense_heatmap.dtype,
                )
                dense_heatmap_i = dense_heatmap * dense_box_classes_i
                dense_box_3d_i = (
                    dense_box_3d * dense_box_classes_i[..., tf.newaxis]
                )
                # Remove z-dimension if this is 2D setup.
                if self._voxel_size[2] > INF_VOXEL_SIZE:
                    dense_heatmap_i = tf.squeeze(dense_heatmap_i, axis=-1)
                    dense_box_3d_i = tf.squeeze(dense_box_3d_i, axis=-2)

                top_k_heatmap_feature_idx_i = None
                if self._top_k_heatmap[i] > 0:
                    top_k_heatmap_feature_idx_i = compute_top_k_heatmap_idx(
                        dense_heatmap_i, self._top_k_heatmap[i]
                    )

                centernet_targets[f"class_{i+1}"] = {
                    "heatmap": dense_heatmap_i,
                    "boxes": dense_box_3d_i,
                    "top_k_index": top_k_heatmap_feature_idx_i,
                }

            inputs.update(centernet_targets)
            return inputs
