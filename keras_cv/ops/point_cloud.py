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


def get_rank(tensor):
    return tensor.shape.ndims or tf.rank(tensor)


def _center_xyzWHD_to_corner_xyz(boxes):
    """convert from center format to corner format.
    Args:
      boxes: [..., num_boxes, 7] float32 Tensor for 3d boxes in [x, y, z, dx,
        dy, dz, phi].
    Returns:
      corners: [..., num_boxes, 8, 3] float32 Tensor for 3d corners in [x, y, z].
    """
    # relative corners w.r.t to origin point
    # this will return all corners in top-down counter clockwise instead of
    # only left top and bottom right.
    rel_corners = tf.constant(
        [
            [0.5, 0.5, 0.5],  # top
            [-0.5, 0.5, 0.5],  # top
            [-0.5, -0.5, 0.5],  # top
            [0.5, -0.5, 0.5],  # top
            [0.5, 0.5, -0.5],  # bottom
            [-0.5, 0.5, -0.5],  # bottom
            [-0.5, -0.5, -0.5],  # bottom
            [0.5, -0.5, -0.5],  # bottom
        ]
    )

    centers = boxes[..., :3]
    dimensions = boxes[..., 3:6]
    phi_world = boxes[..., 6]
    leading_shapes = boxes.shape.as_list()[:-1]
    cos = tf.cos(phi_world)
    sin = tf.sin(phi_world)
    zero = tf.zeros_like(cos)
    one = tf.ones_like(cos)
    rotations = tf.reshape(
        tf.stack([cos, -sin, zero, sin, cos, zero, zero, zero, one], axis=-1),
        leading_shapes + [3, 3],
    )
    # apply the delta to convert from centers to relative corners format
    rel_corners = tf.einsum("...ni,ji->...nji", dimensions, rel_corners)
    # apply rotation matrix on relative corners
    rel_corners = tf.einsum("...nij,...nkj->...nki", rotations, rel_corners)
    # translate back to absolute corners format
    corners = rel_corners + tf.reshape(centers, leading_shapes + [1, 3])
    return corners


def _is_on_lefthand_side(points, v1, v2):
    """Checks if points lay on a vector direction or to its left.

    Args:
      point: float Tensor of [num_points, 2] of points to check
      v1: float Tensor of [num_points, 2] of starting point of the vector
      v2: float Tensor of [num_points, 2] of ending point of the vector

    Returns:
      a boolean Tensor of [num_points] indicate whether each point is on
      the left of the vector or on the vector direction.
    """
    # Prepare for broadcast: All point operations are on the right,
    # and all v1/v2 operations are on the left. This is faster than left/right
    # under the assumption that we have more points than vertices.
    points_x = points[..., tf.newaxis, :, 0]
    points_y = points[..., tf.newaxis, :, 1]
    v1_x = v1[..., 0, tf.newaxis]
    v2_x = v2[..., 0, tf.newaxis]
    v1_y = v1[..., 1, tf.newaxis]
    v2_y = v2[..., 1, tf.newaxis]
    d1 = (points_y - v1_y) * (v2_x - v1_x)
    d2 = (points_x - v1_x) * (v2_y - v1_y)
    return d1 >= d2


def _box_area(boxes):
    """Compute the area of 2-d boxes.

    Vertices must be ordered counter-clockwise. This function can
    technically handle any kind of convex polygons.

    Args:
      boxes: a float Tensor of [..., 4, 2] of boxes. The last coordinates
        are the four corners of the box and (x, y). The corners must be given in
        counter-clockwise order.
    """
    boxes_roll = tf.roll(boxes, shift=1, axis=-2)
    det = (
        tf.reduce_sum(
            boxes[..., 0] * boxes_roll[..., 1] - boxes[..., 1] * boxes_roll[..., 0],
            axis=-1,
            keepdims=True,
        )
        / 2.0
    )
    return tf.abs(det)


def is_within_box2d(points, boxes):
    """Checks if 3d points are within 2d bounding boxes.
    Currently only xy format is supported.
    This function returns true if points are strictly inside the box or on edge.

    Args:
      points: [num_points, 2] float32 Tensor for 2d points in xy format.
      boxes: [num_boxes, 4, 2] float32 Tensor for 2d boxes in xy format,
        counter clockwise.

    Returns:
      boolean Tensor of shape [num_points, num_boxes]
    """
    v1, v2, v3, v4 = (
        boxes[..., 0, :],
        boxes[..., 1, :],
        boxes[..., 2, :],
        boxes[..., 3, :],
    )
    is_inside = tf.math.logical_and(
        tf.math.logical_and(
            _is_on_lefthand_side(points, v1, v2), _is_on_lefthand_side(points, v2, v3)
        ),
        tf.math.logical_and(
            _is_on_lefthand_side(points, v3, v4), _is_on_lefthand_side(points, v4, v1)
        ),
    )
    valid_area = tf.greater(_box_area(boxes), 0)
    is_inside = tf.math.logical_and(is_inside, valid_area)
    # swap the last two dimensions
    is_inside = tf.einsum("...ij->...ji", tf.cast(is_inside, tf.int32))
    return tf.cast(is_inside, tf.bool)


def is_within_box3d(points, boxes):
    """Checks if 3d points are within 3d bounding boxes.
    Currently only xyz format is supported.

    Args:
      points: [..., num_points, 3] float32 Tensor for 3d points in xyz format.
      boxes: [..., num_boxes, 7] float32 Tensor for 3d boxes in [x, y, z, dx,
        dy, dz, phi].

    Returns:
      boolean Tensor of shape [..., num_points, num_boxes] indicating whether
      the point belongs to the box.

    """
    # step 1 -- determine if points are within xy range

    # convert from center format to corner format
    boxes_corner = _center_xyzWHD_to_corner_xyz(boxes)
    # project to 2d boxes by only taking x, y on top plane
    boxes_2d = boxes_corner[..., 0:4, 0:2]
    # project to 2d points by only taking x, y
    points_2d = points[..., :2]
    # check whether points are within 2d boxes, [..., num_points, num_boxes]
    is_inside_2d = is_within_box2d(points_2d, boxes_2d)

    # step 2 -- determine if points are within z range

    [_, _, z, _, _, dz, _] = tf.split(boxes, 7, axis=-1)
    z = z[..., 0]
    dz = dz[..., 0]
    bottom = z - dz / 2.0
    # [..., 1, num_boxes]
    bottom = bottom[..., tf.newaxis, :]
    top = z + dz / 2.0
    top = top[..., tf.newaxis, :]
    # [..., num_points, 1]
    points_z = points[..., 2:]
    # [..., num_points, num_boxes]
    is_inside_z = tf.math.logical_and(
        tf.less_equal(points_z, top), tf.greater_equal(points_z, bottom)
    )
    return tf.math.logical_and(is_inside_z, is_inside_2d)
