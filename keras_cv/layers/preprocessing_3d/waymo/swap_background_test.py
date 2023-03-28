# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/LICENSE
# and https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/PATENTS

import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing_3d.swap_background import SwapBackground

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
ADDITIONAL_POINT_CLOUDS = base_augmentation_layer_3d.ADDITIONAL_POINT_CLOUDS
ADDITIONAL_BOUNDING_BOXES = base_augmentation_layer_3d.ADDITIONAL_BOUNDING_BOXES


class SwapBackgroundTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = SwapBackground()
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4],
                    [10, 1, 2, 3, 4],
                    [0, -1, 2, 3, 4],
                    [100, 100, 2, 3, 4],
                    [20, 20, 21, 1, 0],
                ]
            ]
            * 2
        ).astype("float32")
        # bounding_boxes: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of boxes, num of box features].
        # The first 8 features are [x, y, z, dx, dy, dz, phi, box class].
        bounding_boxes = np.array(
            [
                [
                    [0, 0, 0, 4, 4, 4, 0, 1],
                    [20, 20, 20, 1, 1, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        additional_point_clouds = np.array(
            [
                [
                    [0, 2, 1, 3, 4],
                    [0, 0, 2, 0, 2],
                    [0, 11, 2, 3, 4],
                    [100, 101, 2, 3, 4],
                    [10, 10, 10, 10, 10],
                ]
            ]
            * 2
        ).astype("float32")
        additional_bounding_boxes = np.array(
            [
                [
                    [0, 0, 1, 4, 4, 4, 0, 1],
                    [100, 100, 2, 5, 5, 5, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
            ADDITIONAL_POINT_CLOUDS: additional_point_clouds,
            ADDITIONAL_BOUNDING_BOXES: additional_bounding_boxes,
        }
        outputs = add_layer(inputs)
        # The following points in additional_point_clouds.
        # [0, 2, 1, 3, 4], -> kept because it is in additional_point_clouds [0, 0, 1, 4, 4, 4, 0, 1].
        # [0, 0, 2, 0, 2] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [0, 11, 2, 3, 4] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [100, 101, 2, 3, 4] -> kept because it is in additional_point_clouds [100, 100, 2, 5, 5, 5, 0, 1].
        # [10, 10, 10, 10, 10] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # The following points in point_clouds.
        # [0, 1, 2, 3, 4] -> removed because it is in bounding_boxes [0, 0, 0, 4, 4, 4, 0, 1].
        # [10, 1, 2, 3, 4] -> kept because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [0, -1, 2, 3, 4] -> removed becuase it overlaps with additional_bounding_boxes [0, 0, 1, 4, 4, 4, 0, 1].
        # [100, 100, 2, 3, 4] -> removed becuase it overlaps with additional_bounding_boxes [100, 100, 2, 5, 5, 5, 0, 1].
        # [20, 20, 21, 1, 0] -> kept because it is a background point (not in any bounding_boxes and additional_point_clouds).
        augmented_point_clouds = np.array(
            [
                [
                    [0, 2, 1, 3, 4],
                    [100, 101, 2, 3, 4],
                    [10, 1, 2, 3, 4],
                    [20, 20, 21, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [0, 0, 1, 4, 4, 4, 0, 1],
                    [100, 100, 2, 5, 5, 5, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        self.assertAllClose(
            inputs[ADDITIONAL_POINT_CLOUDS], outputs[ADDITIONAL_POINT_CLOUDS]
        )
        self.assertAllClose(
            inputs[ADDITIONAL_BOUNDING_BOXES],
            outputs[ADDITIONAL_BOUNDING_BOXES],
        )
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = SwapBackground()
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [
                        [0, 1, 2, 3, 4],
                        [10, 1, 2, 3, 4],
                        [0, -1, 2, 3, 4],
                        [100, 100, 2, 3, 4],
                        [20, 20, 21, 1, 0],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        # bounding_boxes: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of boxes, num of box features].
        # The first 8 features are [x, y, z, dx, dy, dz, phi, box class].
        bounding_boxes = np.array(
            [
                [
                    [
                        [0, 0, 0, 4, 4, 4, 0, 1],
                        [20, 20, 20, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        additional_point_clouds = np.array(
            [
                [
                    [
                        [0, 2, 1, 3, 4],
                        [0, 0, 2, 0, 2],
                        [0, 11, 2, 3, 4],
                        [100, 101, 2, 3, 4],
                        [10, 10, 10, 10, 10],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        additional_bounding_boxes = np.array(
            [
                [
                    [
                        [0, 0, 1, 4, 4, 4, 0, 1],
                        [100, 100, 2, 5, 5, 5, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
            ADDITIONAL_POINT_CLOUDS: additional_point_clouds,
            ADDITIONAL_BOUNDING_BOXES: additional_bounding_boxes,
        }
        outputs = add_layer(inputs)
        # The following points in additional_point_clouds.
        # [0, 2, 1, 3, 4], -> kept because it is in additional_point_clouds [0, 0, 1, 4, 4, 4, 0, 1].
        # [0, 0, 2, 0, 2] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [0, 11, 2, 3, 4] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [100, 101, 2, 3, 4] -> kept because it is in additional_point_clouds [100, 100, 2, 5, 5, 5, 0, 1].
        # [10, 10, 10, 10, 10] -> removed because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # The following points in point_clouds.
        # [0, 1, 2, 3, 4] -> removed because it is in bounding_boxes [0, 0, 0, 4, 4, 4, 0, 1].
        # [10, 1, 2, 3, 4] -> kept because it is a background point (not in any bounding_boxes and additional_point_clouds).
        # [0, -1, 2, 3, 4] -> removed becuase it overlaps with additional_bounding_boxes [0, 0, 1, 4, 4, 4, 0, 1].
        # [100, 100, 2, 3, 4] -> removed becuase it overlaps with additional_bounding_boxes [100, 100, 2, 5, 5, 5, 0, 1].
        # [20, 20, 21, 1, 0] -> kept because it is a background point (not in any bounding_boxes and additional_point_clouds).
        augmented_point_clouds = np.array(
            [
                [
                    [
                        [0, 2, 1, 3, 4],
                        [100, 101, 2, 3, 4],
                        [10, 1, 2, 3, 4],
                        [20, 20, 21, 1, 0],
                        [0, 0, 0, 0, 0],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [
                        [0, 0, 1, 4, 4, 4, 0, 1],
                        [100, 100, 2, 5, 5, 5, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2
            ]
            * 3
        ).astype("float32")
        self.assertAllClose(
            inputs[ADDITIONAL_POINT_CLOUDS], outputs[ADDITIONAL_POINT_CLOUDS]
        )
        self.assertAllClose(
            inputs[ADDITIONAL_BOUNDING_BOXES],
            outputs[ADDITIONAL_BOUNDING_BOXES],
        )
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)
