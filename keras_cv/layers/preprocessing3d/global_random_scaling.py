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

from keras_cv.bounding_box_3d import CENTER_XYZ_DXDYDZ_PHI
from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GlobalRandomScaling(base_augmentation_layer_3d.BaseAugmentationLayer3D):
    """A preprocessing layer which randomly scales point clouds and bounding boxes along
    X, Y, and Z axes during training.

    This layer will randomly scale the whole scene along the  X, Y, and Z axes based on a randomly sampled
    scaling factor between [min_scaling_factor, max_scaling_factor] following a uniform distribution.
    During inference time, the output will be identical to input. Call the layer with `training=True` to scale the input.

    Input shape:
      point_clouds: 3D (multi frames) float32 Tensor with shape
        [num of frames, num of points, num of point features].
        The first 5 features are [x, y, z, class, range].
      bounding_boxes:  3D (multi frames) float32 Tensor with shape
        [num of frames, num of boxes, num of box features]. Boxes are expected
        to follow the CENTER_XYZ_DXDYDZ_PHI format. Refer to
        https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py
        for more details on supported bounding box formats.

    Output shape:
      A dictionary of Tensors with the same shape as input Tensors.

    Arguments:
      scaling_factor_x: A tuple of float scalars or a float scalar sets the minimum and maximum scaling factors for the X axis.
      scaling_factor_y: A tuple of float scalars or a float scalar sets the minimum and maximum scaling factors for the Y axis.
      scaling_factor_z: A tuple of float scalars or a float scalar sets the minimum and maximum scaling factors for the Z axis.
    """

    def __init__(
        self,
        scaling_factor_x=None,
        scaling_factor_y=None,
        scaling_factor_z=None,
        same_scaling_xyz=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not scaling_factor_x:
            min_scaling_factor_x = 1.0
            max_scaling_factor_x = 1.0
        elif type(scaling_factor_x) is float:
            min_scaling_factor_x = scaling_factor_x
            max_scaling_factor_x = scaling_factor_x
        else:
            min_scaling_factor_x = scaling_factor_x[0]
            max_scaling_factor_x = scaling_factor_x[1]
        if not scaling_factor_y:
            min_scaling_factor_y = 1.0
            max_scaling_factor_y = 1.0
        elif type(scaling_factor_y) is float:
            min_scaling_factor_y = scaling_factor_y
            max_scaling_factor_y = scaling_factor_y
        else:
            min_scaling_factor_y = scaling_factor_y[0]
            max_scaling_factor_y = scaling_factor_y[1]
        if not scaling_factor_z:
            min_scaling_factor_z = 1.0
            max_scaling_factor_z = 1.0
        elif type(scaling_factor_z) is float:
            min_scaling_factor_z = scaling_factor_z
            max_scaling_factor_z = scaling_factor_z
        else:
            min_scaling_factor_z = scaling_factor_z[0]
            max_scaling_factor_z = scaling_factor_z[1]

        if (
            min_scaling_factor_x < 0
            or max_scaling_factor_x < 0
            or min_scaling_factor_y < 0
            or max_scaling_factor_y < 0
            or min_scaling_factor_z < 0
            or max_scaling_factor_z < 0
        ):
            raise ValueError("min_scaling_factor and max_scaling_factor must be >=0.")
        if (
            min_scaling_factor_x > max_scaling_factor_x
            or min_scaling_factor_y > max_scaling_factor_y
            or min_scaling_factor_z > max_scaling_factor_z
        ):
            raise ValueError("min_scaling_factor must be less than max_scaling_factor.")
        if same_scaling_xyz:
            if (
                min_scaling_factor_x != min_scaling_factor_y
                or min_scaling_factor_y != min_scaling_factor_z
            ):
                raise ValueError(
                    "min_scaling_factor must be the same when same_scaling_xyz is true."
                )
            if (
                max_scaling_factor_x != max_scaling_factor_y
                or max_scaling_factor_y != max_scaling_factor_z
            ):
                raise ValueError(
                    "max_scaling_factor must be the same when same_scaling_xyz is true."
                )

        self._min_scaling_factor_x = min_scaling_factor_x
        self._max_scaling_factor_x = max_scaling_factor_x
        self._min_scaling_factor_y = min_scaling_factor_y
        self._max_scaling_factor_y = max_scaling_factor_y
        self._min_scaling_factor_z = min_scaling_factor_z
        self._max_scaling_factor_z = max_scaling_factor_z
        self._same_scaling_xyz = same_scaling_xyz

    def get_config(self):
        return {
            "scaling_factor_x": (
                self._min_scaling_factor_x,
                self._max_scaling_factor_x,
            ),
            "scaling_factor_y": (
                self._min_scaling_factor_y,
                self._max_scaling_factor_y,
            ),
            "scaling_factor_z": (
                self._min_scaling_factor_z,
                self._max_scaling_factor_z,
            ),
            "same_scaling_xyz": self._same_scaling_xyz,
        }

    def get_random_transformation(self, **kwargs):

        random_scaling_x = self._random_generator.random_uniform(
            (), minval=self._min_scaling_factor_x, maxval=self._max_scaling_factor_x
        )
        random_scaling_y = self._random_generator.random_uniform(
            (), minval=self._min_scaling_factor_y, maxval=self._max_scaling_factor_y
        )
        random_scaling_z = self._random_generator.random_uniform(
            (), minval=self._min_scaling_factor_z, maxval=self._max_scaling_factor_z
        )
        if not self._same_scaling_xyz:
            return {
                "scale": tf.stack(
                    [random_scaling_x, random_scaling_y, random_scaling_z]
                )
            }
        else:
            return {
                "scale": tf.stack(
                    [random_scaling_x, random_scaling_x, random_scaling_x]
                )
            }

    def augment_point_clouds_bounding_boxes(
        self, point_clouds, bounding_boxes, transformation, **kwargs
    ):
        scale = transformation["scale"][tf.newaxis, tf.newaxis, :]
        point_clouds_xyz = point_clouds[..., :3] * scale
        point_clouds = tf.concat([point_clouds_xyz, point_clouds[..., 3:]], axis=-1)

        bounding_boxes_xyzdxdydz = bounding_boxes[
            ..., : CENTER_XYZ_DXDYDZ_PHI.DZ + 1
        ] * tf.concat([scale] * 2, axis=-1)
        bounding_boxes = tf.concat(
            [
                bounding_boxes_xyzdxdydz,
                bounding_boxes[..., CENTER_XYZ_DXDYDZ_PHI.PHI :],
            ],
            axis=-1,
        )

        return (point_clouds, bounding_boxes)
