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
from typing import List
from typing import Sequence
from typing import Union

import numpy as np
import tensorflow as tf

EPSILON = 1e-4


def compute_feature_map_ref_xyz(
    voxel_size: Sequence[float],
    spatial_size: Sequence[float],
    global_xyz: tf.Tensor,
) -> tf.Tensor:
    """Computes the offset xyz locations for each feature map pixel.

    Args:
      voxel_size: voxel size.
      spatial_size: the x, y, z boundary of voxels.
      global_xyz: [B, 3] tensor

    Returns:
      [B, H, W, Z, 3] offset locations for each feature map pixel in global
        coordinate.
    """
    voxel_spatial_size = compute_voxel_spatial_size(spatial_size, voxel_size)
    voxel_coord_meshgrid = np.mgrid[
        0 : voxel_spatial_size[0], 0 : voxel_spatial_size[1], 0 : voxel_spatial_size[2]
    ]
    voxel_coord = np.concatenate(voxel_coord_meshgrid[..., np.newaxis], axis=-1)
    # [H, W, Z, 3]
    voxel_coord = tf.constant(voxel_coord, dtype=global_xyz.dtype)
    # [3]
    voxel_origin = tf.cast(
        compute_voxel_origin(spatial_size, voxel_size),
        dtype=global_xyz.dtype,
    )
    # [H, W, Z, 3]
    voxel_coord = voxel_coord + voxel_origin
    # [H, W, Z, 3]
    ref = voxel_coord_to_point(voxel_coord, voxel_size, dtype=global_xyz.dtype)
    # [1, H, W, Z, 3] + [B, 1, 1, 1, 3] -> [B, H, W, Z, 3]
    ref = ref[tf.newaxis, ...] + global_xyz[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    return ref


def compute_voxel_spatial_size(
    spatial_size: Sequence[float], voxel_size: Sequence[float]
) -> List[int]:
    """Computes how many voxels in each dimension are needed.

    Args:
      spatial_size: max/min range in each dim in global coordinate frame.
      voxel_size: voxel size.

    Returns:
      voxel_spatial_size: voxel spatial size.
    """
    dim = len(voxel_size)
    # Compute the range as x_range = xmax - xmin, ymax - ymin, zmax - zmin
    voxel_spatial_size_float = [
        spatial_size[2 * i + 1] - spatial_size[2 * i] for i in range(dim)
    ]
    # voxel_dim_x / x_range
    voxel_spatial_size_float = [
        i / j for i, j in zip(voxel_spatial_size_float, voxel_size)
    ]
    voxel_spatial_size_int = [math.ceil(v - EPSILON) for v in voxel_spatial_size_float]

    return voxel_spatial_size_int


def compute_voxel_origin(
    spatial_size: Sequence[float],
    voxel_size: Sequence[float],
) -> tf.Tensor:
    """Computes voxel origin.

    Args:
      spatial_size: The current location of SDC.
      voxel_size: 1.0 / voxel size.

    Returns:
      voxel_origin: [dim] the voxel origin.
    """
    voxel_origin = spatial_size[::2]
    voxel_origin = tf.constant(
        [o / v for o, v in zip(voxel_origin, voxel_size)], dtype=tf.float32
    )
    voxel_origin = tf.math.round(voxel_origin)
    voxel_origin = tf.cast(voxel_origin, dtype=tf.int32)
    return voxel_origin


def point_to_voxel_coord(
    point_xyz: tf.Tensor, voxel_size: Sequence[float], dtype=tf.int32
) -> tf.Tensor:
    """Computes the voxel coord given points.

    A voxel x represents [(x-0.5) / voxel_size, (x+0.5) / voxel_size)
    in the coordinate system of the input point_xyz.

    Args:
      point_xyz: [..., dim] point xyz coordinates.
      voxel_size: voxel size.
      dtype: the output dtype.

    Returns:
      voxelized coordinates.
    """
    with tf.name_scope("point_to_voxel_coord"):
        point_voxelized = point_xyz / tf.constant(voxel_size, dtype=point_xyz.dtype)
        assert dtype.is_integer or dtype.is_floating, f"{dtype}"
        # Note: tf.round casts float to the nearest integer. If the float is 0.5, it
        # casts it to the nearest even integer.
        point_voxelized_round = tf.math.round(point_voxelized)
        if dtype.is_floating:
            assert dtype == point_xyz.dtype, f"{dtype}"
            return point_voxelized_round
        return tf.cast(point_voxelized_round, dtype=dtype)


def voxel_coord_to_point(
    voxel_coord: tf.Tensor, voxel_size: Sequence[float], dtype=tf.float32
) -> tf.Tensor:
    """Convert voxel coord to expected point in the original coordinate system.

    This is the reverse of point_to_voxel_coord.

    Args:
      voxel_coord: [..., dim] int tensors for coordinate of each voxel.
      voxel_size: voxel size.
      dtype: output point data type.

    Returns:
      point coordinates.
    """
    with tf.name_scope("voxel_coord_to_point"):
        # This simply computes voxel_coord * voxel_size.
        if voxel_coord.dtype != dtype:
            voxel_coord = tf.cast(voxel_coord, dtype=dtype)
        return voxel_coord * tf.constant(voxel_size, dtype=dtype)


def get_yaw_rotation(yaw, name=None):
    """Gets a rotation matrix given yaw only.

    Args:
      yaw: x-rotation in radians. This tensor can be any shape except an empty
        one.
      name: the op name.

    Returns:
      A rotation tensor with the same data type of the input. Its shape is
        [input_shape, 3 ,3].
    """
    with tf.name_scope("GetYawRotation"):
        cos_yaw = tf.cos(yaw)
        sin_yaw = tf.sin(yaw)
        ones = tf.ones_like(yaw)
        zeros = tf.zeros_like(yaw)

        return tf.stack(
            [
                tf.stack([cos_yaw, -1.0 * sin_yaw, zeros], axis=-1),
                tf.stack([sin_yaw, cos_yaw, zeros], axis=-1),
                tf.stack([zeros, zeros, ones], axis=-1),
            ],
            axis=-2,
        )


def inv_loc(rot: tf.Tensor, loc: tf.Tensor) -> tf.Tensor:
    """Invert a location.
    rot and loc can form a transform matrix between two frames.
    R = rot, L = loc
    R*R' = I
    R * new_loc + L = 0 = > new_loc = -R'*L
    Args:
      rot: [..., 3, 3] rotation matrix.
      loc: [..., 3] location matrix.
    Returns:
      [..., 3] new location matrix.
    """
    new_loc = -1.0 * tf.linalg.matmul(rot, loc[..., tf.newaxis], transpose_a=True)
    return tf.squeeze(new_loc, axis=-1)


def shape_int_compatible(t: tf.Tensor) -> tf.TensorShape:
    """int32 and int64 compatible tf shape implementation."""
    # tf.shape int32/int64 requires input and output to be on host.
    dtype = t.dtype
    if dtype == tf.int32:
        t = tf.bitcast(t, tf.float32)
    if dtype == tf.int64:
        t = tf.bitcast(t, tf.float64)

    return tf.shape(t)


def combined_static_and_dynamic_shape(tensor: tf.Tensor) -> List[Union[tf.Tensor, int]]:
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = shape_int_compatible(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def _has_rank(tensor, expected_rank):
    """Syntactic sugar for asserting that tensor has the expected rank.

    Internal usages for keras_cv libraries only.
    """
    if tensor.shape.ndims is not None and isinstance(expected_rank, int):
        assert tensor.shape.ndims == expected_rank, (
            "Ranks did not match, got %d, " "expected %d"
        ) % (tensor.shape.ndims, expected_rank)
    return tensor


def _pad_or_trim_to(x, shape, pad_val=0, pad_after_contents=True):
    """Pad and slice x to the given shape.

    This is branched from Lingvo https://github.com/tensorflow/lingvo/blob/master/lingvo/core/py_utils.py.

    Internal usages for keras_cv libraries only.

    Args:
      x: A tensor.
      shape: The shape of the returned tensor.
      pad_val: An int or float used to pad x.
      pad_after_contents: Whether to pad and trim after the original contents of
        each dimension.
    Returns:
      'x' is padded with pad_val and sliced so that the result has the given
      shape.
    Raises:
      ValueError: if shape is a tf.TensorShape and not fully defined.
    """
    if isinstance(shape, (list, tuple)):
        expected_rank = len(shape)
    elif isinstance(shape, tf.TensorShape):
        if not shape.is_fully_defined():
            raise ValueError("shape %s padding %s must be fully defined." % (shape, x))
        expected_rank = shape.rank
    else:
        shape = _has_rank(shape, 1)
        expected_rank = tf.size(shape)
    x = _has_rank(x, expected_rank)

    pad = shape - tf.minimum(tf.shape(x), shape)
    zeros = tf.zeros_like(pad)
    if pad_after_contents:
        # If dim_i is less than shape[i], pads after contents.
        paddings = tf.stack([zeros, pad], axis=1)
        # If dim_i is larger than shape[i], we slice [0:shape[i]] for dim_i.
        slice_begin = zeros
    else:
        # If dim_i is less than shape[i], pads before contents.
        paddings = tf.stack([pad, zeros], axis=1)
        # If dim-i is larger than shape[i], we slice [dim_i - shape[i]:dim_i]
        # for dim_i.
        slice_begin = tf.shape(x) + pad - shape

    x = tf.pad(x, paddings, constant_values=pad_val)
    x = tf.slice(x, slice_begin, shape)

    return tf.reshape(x, shape)
