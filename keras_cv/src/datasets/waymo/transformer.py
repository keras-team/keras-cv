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
"""Transformer to convert Waymo Open Dataset proto to model inputs."""

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
import tensorflow as tf

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.utils import assert_waymo_open_dataset_installed

try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import box_utils
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
except ImportError:
    waymo_open_dataset = None

from keras_cv.src.datasets.waymo import struct
from keras_cv.src.layers.object_detection_3d import voxel_utils

WOD_FRAME_OUTPUT_SIGNATURE = {
    "frame_id": tf.TensorSpec((), tf.int64),
    "timestamp_offset": tf.TensorSpec((), tf.float32),
    "timestamp_micros": tf.TensorSpec((), tf.int64),
    "pose": tf.TensorSpec([4, 4], tf.float32),
    "point_xyz": tf.TensorSpec([None, 3], tf.float32),
    "point_feature": tf.TensorSpec([None, 4], tf.float32),
    "point_mask": tf.TensorSpec([None], tf.bool),
    "point_range_image_row_col_sensor_id": tf.TensorSpec([None, 3], tf.float32),
    # Please refer to Waymo Open Dataset label proto for definitions.
    "label_box": tf.TensorSpec([None, 7], tf.float32),
    "label_box_id": tf.TensorSpec([None], tf.int64),
    "label_box_meta": tf.TensorSpec([None, 4], tf.float32),
    "label_box_class": tf.TensorSpec([None], tf.int32),
    "label_box_density": tf.TensorSpec([None], tf.int32),
    "label_box_detection_difficulty": tf.TensorSpec([None], tf.int32),
    "label_box_mask": tf.TensorSpec([None], tf.bool),
    "label_point_class": tf.TensorSpec([None], tf.int32),
    "label_point_nlz": tf.TensorSpec([None], tf.int32),
}

# Maximum number of points from all lidars excluding the top lidar. Please refer
# to https://arxiv.org/pdf/1912.04838.pdf Figure 1 for sensor layouts.
_MAX_NUM_NON_TOP_LIDAR_POINTS = 30000


def _decode_range_images(frame) -> Dict[int, List[tf.Tensor]]:
    """Decodes range images from a Waymo Open Dataset frame.

    Please refer to https://arxiv.org/pdf/1912.04838.pdf for more details.

    Args:
      frame: a Waymo Open Dataset frame.

    Returns:
      A dictionary mapping from sensor ID to list of range images ordered by
        return indices.
    """
    range_images = {}
    for lidar in frame.lasers:
        range_image_str_tensor = tf.io.decode_compressed(
            lidar.ri_return1.range_image_compressed, "ZLIB"
        )
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
        ri_tensor = tf.reshape(
            tf.convert_to_tensor(value=ri.data, dtype=tf.float32), ri.shape.dims
        )
        range_images[lidar.name] = [ri_tensor]

        if lidar.name == dataset_pb2.LaserName.TOP:
            range_image_str_tensor = tf.io.decode_compressed(
                lidar.ri_return2.range_image_compressed, "ZLIB"
            )
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            ri_tensor = tf.reshape(
                tf.convert_to_tensor(value=ri.data, dtype=tf.float32),
                ri.shape.dims,
            )
            range_images[lidar.name].append(ri_tensor)
    return range_images


def _get_range_image_top_pose(frame) -> tf.Tensor:
    """Extracts range image pose tensor.

    Args:
      frame: a Waymo Open Dataset frame.

    Returns:
      Pose tensors for the range image.
    """
    _, _, _, ri_pose = frame_utils.parse_range_image_and_camera_projection(
        frame
    )
    assert ri_pose
    ri_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=ri_pose.data), ri_pose.shape.dims
    )
    # [H, W, 3, 3]
    ri_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        ri_pose_tensor[..., 0], ri_pose_tensor[..., 1], ri_pose_tensor[..., 2]
    )
    ri_pose_tensor_translation = ri_pose_tensor[..., 3:]
    ri_pose_tensor = transform_utils.get_transform(
        ri_pose_tensor_rotation, ri_pose_tensor_translation
    )
    return ri_pose_tensor


def _get_point_top_lidar(
    range_image: Sequence[tf.Tensor], frame
) -> struct.PointTensors:
    """Gets point related tensors for the top lidar.

    Please refer to https://arxiv.org/pdf/1912.04838.pdf Table 2 for lidar
    specifications.

    Args:
      range_image: range image tensors. The range image is:
        [range, intensity, elongation, is_in_nlz].
      frame: a Waymo Open Dataset frame.

    Returns:
      Point tensors.
    """
    assert len(range_image) == 2
    xyz_list = []
    feature_list = []
    row_col_list = []
    nlz_list = []
    has_second_return_list = []
    is_second_return_list = []

    # Extracts frame pose tensor.
    frame_pose_tensor = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4])
    )

    # Extracts range image pose tensor.
    ri_pose_tensor = _get_range_image_top_pose(frame)

    # Extracts calibration data.
    calibration = _get_lidar_calibration(frame, dataset_pb2.LaserName.TOP)
    extrinsic = tf.reshape(np.array(calibration.extrinsic.transform), [4, 4])
    beam_inclinations = tf.constant(calibration.beam_inclinations)
    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

    for i in range(2):
        ri_tensor = range_image[i]
        mask = ri_tensor[:, :, 0] > 0
        mask_idx = tf.cast(tf.where(mask), dtype=tf.int32)

        xyz = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(ri_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(beam_inclinations, axis=0),
            pixel_pose=tf.expand_dims(ri_pose_tensor, axis=0),
            frame_pose=tf.expand_dims(frame_pose_tensor, axis=0),
        )
        xyz = tf.gather_nd(tf.squeeze(xyz, axis=0), mask_idx)
        feature = tf.gather_nd(ri_tensor[:, :, 1:3], mask_idx)
        nlz = tf.gather_nd(ri_tensor[:, :, -1] > 0, mask_idx)

        xyz_list.append(xyz)
        feature_list.append(feature)
        nlz_list.append(nlz)
        row_col_list.append(mask_idx)

        if i == 0:
            has_second_return = range_image[1][:, :, 0] > 0
            has_second_return_list.append(
                tf.gather_nd(has_second_return, mask_idx)
            )
            is_second_return_list.append(
                tf.zeros([mask_idx.shape[0]], dtype=tf.bool)
            )
        else:
            has_second_return_list.append(
                tf.zeros([mask_idx.shape[0]], dtype=tf.bool)
            )
            is_second_return_list.append(
                tf.ones([mask_idx.shape[0]], dtype=tf.bool)
            )

    xyz = tf.concat(xyz_list, axis=0)
    feature = tf.concat(feature_list, axis=0)
    row_col = tf.concat(row_col_list, axis=0)
    nlz = tf.concat(nlz_list, axis=0)
    has_second_return = tf.cast(
        tf.concat(has_second_return_list, axis=0), dtype=tf.float32
    )
    is_second_return = tf.cast(
        tf.concat(is_second_return_list, axis=0), dtype=tf.float32
    )
    # Complete feature: intensity, elongation, has_second, is_second.
    feature = tf.concat(
        [
            feature,
            has_second_return[:, tf.newaxis],
            is_second_return[:, tf.newaxis],
        ],
        axis=-1,
    )
    sensor_id = (
        tf.ones([xyz.shape[0], 1], dtype=tf.int32) * dataset_pb2.LaserName.TOP
    )
    ri_row_col_sensor_id = tf.concat([row_col, sensor_id], axis=-1)

    return struct.PointTensors(
        point_xyz=xyz,
        point_feature=feature,
        point_range_image_row_col_sensor_id=ri_row_col_sensor_id,
        label_point_nlz=nlz,
    )


def _get_lidar_calibration(frame, name: int):
    """Gets lidar calibration for a given lidar."""
    calibration = None
    for c in frame.context.laser_calibrations:
        if c.name == name:
            calibration = c
    assert calibration is not None
    return calibration


def _downsample(point: struct.PointTensors, n: int) -> struct.PointTensors:
    """Randomly samples up to n points from the given point_tensor."""
    num_points = point.point_xyz.shape[0]
    if num_points <= n:
        return point
    mask = tf.range(start=0, limit=num_points, dtype=tf.int32)
    mask = tf.random.shuffle(mask)
    mask_index = mask[:n]

    def _gather(t: tf.Tensor) -> tf.Tensor:
        return tf.gather(t, mask_index)

    tensors = {key: _gather(value) for key, value in vars(point).items()}
    return struct.PointTensors(**tensors)


def _get_point_lidar(
    ris: Dict[int, List[tf.Tensor]],
    frame,
    max_num_points: int,
) -> struct.PointTensors:
    """Gets point related tensors for non-top lidar.

    The main differences from top lidar extraction are related to second return
    and point down sampling.

    Args:
      ris: Mapping from lidar ID to range image tensor. The ri format is [range,
        intensity, elongation, is_in_nlz].
      frame: a Waymo Open Dataset frame.
      max_num_points: maximum number of points from non-top lidar.

    Returns:
      Point related tensors.
    """
    xyz_list = []
    feature_list = []
    nlz_list = []
    ri_row_col_sensor_id_list = []

    for sensor_id in ris.keys():
        ri_tensor = ris[sensor_id]
        assert len(ri_tensor) == 1, f"{sensor_id}"
        ri_tensor = ri_tensor[0]
        calibration = _get_lidar_calibration(frame, sensor_id)
        extrinsic = tf.reshape(
            np.array(calibration.extrinsic.transform), [4, 4]
        )
        beam_inclinations = range_image_utils.compute_inclination(
            tf.constant(
                [
                    calibration.beam_inclination_min,
                    calibration.beam_inclination_max,
                ]
            ),
            height=ri_tensor.shape[0],
        )
        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        xyz = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(ri_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(beam_inclinations, axis=0),
        )
        mask = ri_tensor[:, :, 0] > 0
        mask_idx = tf.cast(tf.where(mask), dtype=tf.int32)

        xyz = tf.gather_nd(tf.squeeze(xyz, axis=0), mask_idx)
        feature = tf.gather_nd(ri_tensor[:, :, 1:3], mask_idx)
        feature = tf.concat(
            [feature, tf.zeros([feature.shape[0], 2], dtype=tf.float32)],
            axis=-1,
        )
        nlz = tf.gather_nd(ri_tensor[:, :, -1] > 0, mask_idx)

        xyz_list.append(xyz)
        feature_list.append(feature)
        nlz_list.append(nlz)
        ri_row_col_sensor_id_list.append(
            tf.concat(
                [
                    mask_idx,
                    sensor_id * tf.ones([nlz.shape[0], 1], dtype=tf.int32),
                ],
                axis=-1,
            )
        )

    xyz = tf.concat(xyz_list, axis=0)
    feature = tf.concat(feature_list, axis=0)
    nlz = tf.concat(nlz_list, axis=0)
    ri_row_col_sensor_id = tf.concat(ri_row_col_sensor_id_list, axis=0)

    point_tensors = struct.PointTensors(
        point_xyz=xyz,
        point_feature=feature,
        point_range_image_row_col_sensor_id=ri_row_col_sensor_id,
        label_point_nlz=nlz,
    )
    point_tensors = _downsample(point_tensors, max_num_points)
    return point_tensors


def _get_point(frame, max_num_lidar_points: int) -> struct.PointTensors:
    """Gets point related tensors from a Waymo Open Dataset frame.

    Args:
      frame: a Waymo Open Dataset frame.
      max_num_lidar_points: maximum number of points from non-top lidars.

    Returns:
      Point related tensors.
    """
    range_images = _decode_range_images(frame)
    point_top_lidar = _get_point_top_lidar(
        range_images[dataset_pb2.LaserName.TOP], frame
    )

    range_images.pop(dataset_pb2.LaserName.TOP)
    point_tensors_lidar = _get_point_lidar(
        range_images, frame, max_num_lidar_points
    )

    merged = {}
    for key in vars(point_tensors_lidar).keys():
        merged[key] = tf.concat(
            [getattr(point_tensors_lidar, key), getattr(point_top_lidar, key)],
            axis=0,
        )
    return struct.PointTensors(**merged)


def _get_point_label_box(
    frame,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Extracts 3D box labels from a Waymo Open Dataset frame.

    Args:
      frame: a Waymo Open Dataset frame.

    Returns:
      box_3d: [M, 7] 3d boxes.
      box_meta: [M, 4] speed and accel for each box.
      box_class: [M] object class of each box.
      box_id: [M] unique ID of each box.
      box_density: [M] number of points in each box.
      box_detection_difficulty: [M] difficulty level for detection.
    """
    box_3d_list = []
    box_meta_list = []
    box_class_list = []
    box_id_list = []
    box_density_list = []
    box_detection_difficulty_list = []

    for label in frame.laser_labels:
        model_object_type = label.type
        density = label.num_lidar_points_in_box
        detection_difficulty = label.detection_difficulty_level

        if model_object_type == 0:
            continue
        b = label.box
        box_3d_list.extend(
            [
                b.center_x,
                b.center_y,
                b.center_z,
                b.length,
                b.width,
                b.height,
                b.heading,
            ]
        )
        meta = label.metadata
        box_meta_list.extend(
            [
                meta.speed_x,
                meta.speed_y,
                meta.accel_x,
                meta.accel_y,
            ]
        )
        box_class_list.append(model_object_type)
        box_id = tf.bitcast(
            tf.fingerprint(
                tf.expand_dims(label.id.encode(encoding="ascii"), 0)
            )[0],
            tf.int64,
        )
        box_id_list.append(box_id)
        box_density_list.append(density)
        box_detection_difficulty_list.append(detection_difficulty)

    box_3d = tf.reshape(tf.constant(box_3d_list, dtype=tf.float32), [-1, 7])
    box_meta = tf.reshape(tf.constant(box_meta_list, dtype=tf.float32), [-1, 4])
    box_class = tf.constant(box_class_list, dtype=tf.int32)
    box_id = tf.stack(box_id_list)
    box_density = tf.constant(box_density_list, dtype=tf.int32)
    box_detection_difficulty = tf.constant(
        box_detection_difficulty_list, dtype=tf.int32
    )
    return (
        box_3d,
        box_meta,
        box_class,
        box_id,
        box_density,
        box_detection_difficulty,
    )


def _get_box_class_per_point(
    box: tf.Tensor, box_class: tf.Tensor, point_xyz: tf.Tensor
) -> tf.Tensor:
    """Extracts point labels.

    Args:
      box: [M, 7] box tensor.
      box_class: [M] class of each box.
      point_xyz: [N, 3] points.

    Returns:
      point_box_class: [N] box class of each point.
    """
    n = point_xyz.shape[0]
    m = box.shape[0]
    if m == 0:
        return tf.zeros([n], dtype=tf.int32)

    # [N, M]
    point_in_box = box_utils.is_within_box_3d(point_xyz, box)
    # [N]
    point_in_any_box = tf.math.reduce_any(point_in_box, axis=-1)
    # [N]
    point_box_idx = tf.math.argmax(point_in_box, axis=-1, output_type=tf.int32)
    # [N]
    point_box_class = tf.where(
        point_in_any_box, tf.gather(box_class, point_box_idx), 0
    )

    return point_box_class


def _get_point_label(frame, point_xyz: tf.Tensor) -> struct.LabelTensors:
    """Extracts labels.

    Args:
      frame: an open dataset frame.
      point_xyz: [N, 3] tensor representing point xyz.

    Returns:
      Label tensors.
    """
    (
        box_3d,
        box_meta,
        box_class,
        box_id,
        box_density,
        box_detection_difficulty,
    ) = _get_point_label_box(frame)
    point_box_class = _get_box_class_per_point(box_3d, box_class, point_xyz)
    box_mask = tf.math.greater(box_class, 0)
    return struct.LabelTensors(
        label_box=box_3d,
        label_box_id=box_id,
        label_box_meta=box_meta,
        label_box_class=box_class,
        label_box_density=box_density,
        label_box_detection_difficulty=box_detection_difficulty,
        label_box_mask=box_mask,
        label_point_class=point_box_class,
    )


def _point_vehicle_to_global(
    point_vehicle_xyz: tf.Tensor, sdc_pose: tf.Tensor
) -> tf.Tensor:
    """Transforms points from vehicle to global frame.

    Args:
      point_vehicle_xyz: [..., N, 3] vehicle xyz.
      sdc_pose: [..., 4, 4] the SDC pose.

    Returns:
      The points in global frame.
    """
    rot = sdc_pose[..., 0:3, 0:3]
    loc = sdc_pose[..., 0:3, 3]
    return (
        tf.linalg.matmul(point_vehicle_xyz, rot, transpose_b=True)
        + loc[..., tf.newaxis, :]
    )


def _point_global_to_vehicle(
    point_xyz: tf.Tensor, sdc_pose: tf.Tensor
) -> tf.Tensor:
    """Transforms points from global to vehicle frame.

    Args:
      point_xyz: [..., N, 3] global xyz.
      sdc_pose: [..., 4, 4] the SDC pose.

    Returns:
      The points in vehicle frame.
    """
    rot = sdc_pose[..., 0:3, 0:3]
    loc = sdc_pose[..., 0:3, 3]
    return (
        tf.linalg.matmul(point_xyz, rot)
        + voxel_utils.inv_loc(rot, loc)[..., tf.newaxis, :]
    )


def _box_3d_vehicle_to_global(
    box_3d: tf.Tensor, sdc_pose: tf.Tensor
) -> tf.Tensor:
    """Transforms 3D boxes from vehicle to global frame.

    Args:
      box_3d: [..., N, 7] 3d boxes in vehicle frame.
      sdc_pose: [..., 4, 4] the SDC pose.

    Returns:
      The boxes in global frame.
    """
    center = box_3d[..., 0:3]
    dim = box_3d[..., 3:6]
    heading = box_3d[..., 6]

    new_center = _point_vehicle_to_global(center, sdc_pose)
    new_heading = (
        heading
        + tf.atan2(sdc_pose[..., 1, 0], sdc_pose[..., 0, 0])[..., tf.newaxis]
    )

    return tf.concat([new_center, dim, new_heading[..., tf.newaxis]], axis=-1)


def _box_3d_global_to_vehicle(
    box_3d: tf.Tensor, sdc_pose: tf.Tensor
) -> tf.Tensor:
    """Transforms 3D boxes from global to vehicle frame.

    Args:
      box_3d: [..., N, 7] 3d boxes in global frame.
      sdc_pose: [..., 4, 4] the SDC pose.

    Returns:
      The boxes in vehicle frame.
    """
    center = box_3d[..., 0:3]
    dim = box_3d[..., 3:6]
    heading = box_3d[..., 6]

    new_center = _point_global_to_vehicle(center, sdc_pose)
    new_heading = (
        heading
        + tf.atan2(sdc_pose[..., 0, 1], sdc_pose[..., 0, 0])[..., tf.newaxis]
    )

    return tf.concat([new_center, dim, new_heading[..., tf.newaxis]], axis=-1)


@keras_cv_export("keras_cv.datasets.waymo.build_tensors_from_wod_frame")
def build_tensors_from_wod_frame(frame) -> Dict[str, tf.Tensor]:
    """Builds tensors from a Waymo Open Dataset frame.

    This function is to convert range image to point cloud. User can also work
    with range image directly with frame_utils functions from
    waymo_open_dataset.

    Args:
      frame: a Waymo Open Dataset frame.

    Returns:
      Flat dictionary of tensors.
    """
    assert_waymo_open_dataset_installed(
        "keras_cv.datasets.waymo.build_tensors_from_wod_frame()"
    )

    frame_id_bytes = "{}_{}".format(
        frame.context.name, frame.timestamp_micros
    ).encode(encoding="ascii")
    frame_id = tf.bitcast(
        tf.fingerprint(tf.expand_dims(frame_id_bytes, 0))[0], tf.int64
    )

    timestamp_micros = tf.constant(frame.timestamp_micros, dtype=tf.int64)
    pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]),
        dtype_hint=tf.float32,
    )

    point_tensors = _get_point(frame, _MAX_NUM_NON_TOP_LIDAR_POINTS)
    point_label_tensors = _get_point_label(frame, point_tensors.point_xyz)

    # Transforms lidar frames to global coordinates.
    point_tensors.point_xyz = _point_vehicle_to_global(
        point_tensors.point_xyz, pose
    )
    point_label_tensors.label_box = _box_3d_vehicle_to_global(
        point_label_tensors.label_box, pose
    )

    # Constructs final results.
    num_points = point_tensors.point_xyz.shape[0]
    return {
        "frame_id": frame_id,
        "timestamp_offset": tf.constant(0.0, dtype=tf.float32),
        "timestamp_micros": timestamp_micros,
        "pose": pose,
        "point_xyz": point_tensors.point_xyz,
        "point_feature": point_tensors.point_feature,
        "point_mask": tf.ones([num_points], dtype=tf.bool),
        "point_range_image_row_col_sensor_id": point_tensors.point_range_image_row_col_sensor_id,  # noqa: E501
        "label_box": point_label_tensors.label_box,
        "label_box_id": point_label_tensors.label_box_id,
        "label_box_meta": point_label_tensors.label_box_meta,
        "label_box_class": point_label_tensors.label_box_class,
        "label_box_density": point_label_tensors.label_box_density,
        "label_box_detection_difficulty": point_label_tensors.label_box_detection_difficulty,  # noqa: E501
        "label_box_mask": point_label_tensors.label_box_mask,
        "label_point_class": point_label_tensors.label_point_class,
        "label_point_nlz": point_tensors.label_point_nlz,
    }


@keras_cv_export("keras_cv.datasets.waymo.pad_or_trim_tensors")
def pad_or_trim_tensors(
    frame: Dict[str, tf.Tensor], max_num_point=199600, max_num_label_box=1000
) -> Dict[str, tf.Tensor]:
    """Pad or trim tensors from a frame to have uniform shapes.

    Args:
      frame: a dictionary of feature tensors from a Waymo Open Dataset frame.
      max_num_point: maximum number of lidar points to process.
      max_num_label_box: maximum number of label boxes to process.


    Returns:
      A dictionary of feature tensors with uniform shapes.
    """

    def _pad_fn(t: tf.Tensor, max_counts: int) -> tf.Tensor:
        shape = [max_counts] + t.shape.as_list()[1:]
        return voxel_utils._pad_or_trim_to(t, shape)

    point_tensor_keys = {
        "point_xyz",
        "point_feature",
        "point_range_image_row_col_sensor_id",
        "point_mask",
        "label_point_class",
        "label_point_nlz",
    }
    box_tensor_keys = {
        "label_box",
        "label_box_id",
        "label_box_meta",
        "label_box_class",
        "label_box_density",
        "label_box_detection_difficulty",
        "label_box_mask",
    }
    for key in point_tensor_keys:
        t = frame[key]
        if t is not None:
            frame[key] = _pad_fn(t, max_num_point)
    for key in box_tensor_keys:
        t = frame[key]
        if t is not None:
            frame[key] = _pad_fn(t, max_num_label_box)
    return frame


@keras_cv_export("keras_cv.datasets.waymo.transform_to_vehicle_frame")
def transform_to_vehicle_frame(
    frame: Dict[str, tf.Tensor]
) -> Dict[str, tf.Tensor]:
    """Transform tensors in a frame from global coordinates to vehicle
    coordinates.

    Args:
      frame: a dictionary of feature tensors from a Waymo Open Dataset frame in
        global frame.


    Returns:
      A dictionary of feature tensors in vehicle frame.
    """
    assert_waymo_open_dataset_installed(
        "keras_cv.datasets.waymo.transform_to_vehicle_frame()"
    )

    def _transform_to_vehicle_frame(
        point_global_xyz: tf.Tensor,
        point_mask: tf.Tensor,
        box_global: tf.Tensor,
        box_mask: tf.Tensor,
        sdc_pose: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        point_vehicle_xyz = _point_global_to_vehicle(point_global_xyz, sdc_pose)
        point_vehicle_xyz = tf.where(
            point_mask[..., tf.newaxis], point_vehicle_xyz, 0.0
        )
        box_vehicle = _box_3d_global_to_vehicle(box_global, sdc_pose)
        box_vehicle = tf.where(box_mask[..., tf.newaxis], box_vehicle, 0.0)
        return point_vehicle_xyz, box_vehicle

    point_vehicle_xyz, box_vehicle = _transform_to_vehicle_frame(
        frame["point_xyz"],
        frame["point_mask"],
        frame["label_box"],
        frame["label_box_mask"],
        frame["pose"],
    )
    frame["point_xyz"] = point_vehicle_xyz
    frame["label_box"] = box_vehicle
    # Override pose as the points and boxes are in the vehicle frame.
    frame["pose"] = tf.eye(4)
    if frame["label_point_nlz"] is not None:
        frame["point_mask"] = tf.logical_and(
            frame["point_mask"],
            tf.logical_not(tf.cast(frame["label_point_nlz"], tf.bool)),
        )
    return frame


@keras_cv_export("keras_cv.datasets.waymo.convert_to_center_pillar_inputs")
def convert_to_center_pillar_inputs(
    frame: Dict[str, tf.Tensor]
) -> Dict[str, Any]:
    """Converts an input frame into CenterPillar input format.

    Args:
      frame: a dictionary of feature tensors from a Waymo Open Dataset frame

    Returns:
      A dictionary of two tensor dictionaries with keys "point_clouds"
      and "3d_boxes".
    """
    point_clouds = {
        "point_xyz": frame["point_xyz"],
        "point_feature": frame["point_feature"],
        "point_mask": frame["point_mask"],
    }
    boxes = {
        "boxes": frame["label_box"],
        "classes": frame["label_box_class"],
        "difficulty": frame["label_box_detection_difficulty"],
        "mask": frame["label_box_mask"],
    }
    y = {
        "point_clouds": point_clouds,
        "3d_boxes": boxes,
    }
    return y


@keras_cv_export("keras_cv.datasets.waymo.build_tensors_for_augmentation")
def build_tensors_for_augmentation(
    frame: Dict[str, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Builds tensors for data augmentation from an input frame.

    Args:
      frame: a dictionary of feature tensors from a Waymo Open Dataset frame

    Returns:
      A dictionary of two tensors with keys "point_clouds" and "bounding_boxes"
      and values which are tensors of shapes [num points, num features] and
      [num boxes, num features]).
    """
    assert_waymo_open_dataset_installed(
        "keras_cv.datasets.waymo.build_tensors_for_augmentation()"
    )
    point_cloud = tf.concat(
        [
            frame["point_xyz"][tf.newaxis, ...],
            frame["point_feature"][tf.newaxis, ...],
            tf.cast(frame["point_mask"], tf.float32)[tf.newaxis, :, tf.newaxis],
        ],
        axis=-1,
    )
    boxes = tf.concat(
        [
            frame["label_box"][tf.newaxis, :],
            tf.cast(frame["label_box_class"], tf.float32)[
                tf.newaxis, :, tf.newaxis
            ],
            tf.cast(frame["label_box_mask"], tf.float32)[
                tf.newaxis, :, tf.newaxis
            ],
            tf.cast(frame["label_box_density"], tf.float32)[
                tf.newaxis, :, tf.newaxis
            ],
            tf.cast(frame["label_box_detection_difficulty"], tf.float32)[
                tf.newaxis, :, tf.newaxis
            ],
        ],
        axis=-1,
    )
    return {
        "point_clouds": tf.squeeze(point_cloud, axis=0),
        "bounding_boxes": tf.squeeze(boxes, axis=0),
    }
