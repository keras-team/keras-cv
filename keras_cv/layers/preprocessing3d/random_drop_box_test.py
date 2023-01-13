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
import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing3d.random_drop_box import RandomDropBox

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES
ADDITIONAL_POINT_CLOUDS = base_augmentation_layer_3d.ADDITIONAL_POINT_CLOUDS
ADDITIONAL_BOUNDING_BOXES = base_augmentation_layer_3d.ADDITIONAL_BOUNDING_BOXES


class RandomDropBoxTest(tf.test.TestCase):
    def test_drop_class1_box_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(2)
        add_layer = RandomDropBox(label_index=1, max_drop_bounding_boxes=4)
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4],
                    [0, 0, 2, 3, 4],
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
                    [20, 20, 20, 1, 1, 1, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")

        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
        }
        outputs = add_layer(inputs)
        # Drop the first object bounding box [0, 0, 0, 4, 4, 4, 0, 1] and points.
        augmented_point_clouds = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [10, 1, 2, 3, 4],
                    [0, 0, 0, 0, 0],
                    [100, 100, 2, 3, 4],
                    [20, 20, 21, 1, 0],
                ]
            ]
            * 2
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [20, 20, 20, 1, 1, 1, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)

    def test_drop_both_boxes_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(2)
        add_layer = RandomDropBox(max_drop_bounding_boxes=4)
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4],
                    [0, 0, 2, 3, 4],
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
                    [20, 20, 20, 3, 3, 3, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")

        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
        }
        outputs = add_layer(inputs)
        # Drop both object bounding boxes and points.
        augmented_point_clouds = np.array(
            [
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [10, 1, 2, 3, 4],
                    [0, 0, 0, 0, 0],
                    [100, 100, 2, 3, 4],
                    [0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)

    def test_not_drop_any_box_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(2)
        add_layer = RandomDropBox(max_drop_bounding_boxes=0)
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4],
                    [0, 0, 2, 3, 4],
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
                    [20, 20, 20, 3, 3, 3, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")

        inputs = {
            POINT_CLOUDS: point_clouds,
            BOUNDING_BOXES: bounding_boxes,
        }
        outputs = add_layer(inputs)
        # Do not drop any bounding box or point.
        augmented_point_clouds = np.array(
            [
                [
                    [0, 1, 2, 3, 4],
                    [0, 0, 2, 3, 4],
                    [10, 1, 2, 3, 4],
                    [0, -1, 2, 3, 4],
                    [100, 100, 2, 3, 4],
                    [20, 20, 21, 1, 0],
                ]
            ]
            * 2
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [0, 0, 0, 4, 4, 4, 0, 1],
                    [20, 20, 20, 3, 3, 3, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
            * 2
        ).astype("float32")
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)

    def test_batch_drop_one_of_the_box_point_clouds_and_bounding_boxes(self):
        tf.keras.utils.set_random_seed(4)
        add_layer = RandomDropBox(max_drop_bounding_boxes=2)
        # point_clouds: 3D (multi frames) float32 Tensor with shape
        # [num of frames, num of points, num of point features].
        # The first 5 features are [x, y, z, class, range].
        point_clouds = np.array(
            [
                [
                    [
                        [0, 1, 2, 3, 4],
                        [0, 0, 2, 3, 4],
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
                        [20, 20, 20, 3, 3, 3, 0, 2],
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
        }
        outputs = add_layer(inputs)
        # Batch 0: drop the first bounding box [0, 0, 0, 4, 4, 4, 0, 1] and points,
        # Batch 1,2: drop the second bounding box [20, 20, 20, 3, 3, 3, 0, 2] and points,
        augmented_point_clouds = np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [10, 1, 2, 3, 4],
                        [0, 0, 0, 0, 0],
                        [100, 100, 2, 3, 4],
                        [20, 20, 21, 1, 0],
                    ]
                ]
                * 2,
                [
                    [
                        [0, 1, 2, 3, 4],
                        [0, 0, 2, 3, 4],
                        [10, 1, 2, 3, 4],
                        [0, -1, 2, 3, 4],
                        [100, 100, 2, 3, 4],
                        [0, 0, 0, 0, 0],
                    ]
                ]
                * 2,
                [
                    [
                        [0, 1, 2, 3, 4],
                        [0, 0, 2, 3, 4],
                        [10, 1, 2, 3, 4],
                        [0, -1, 2, 3, 4],
                        [100, 100, 2, 3, 4],
                        [0, 0, 0, 0, 0],
                    ]
                ]
                * 2,
            ]
        ).astype("float32")
        augmented_bounding_boxes = np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [20, 20, 20, 3, 3, 3, 0, 2],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2,
                [
                    [
                        [0, 0, 0, 4, 4, 4, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2,
                [
                    [
                        [0, 0, 0, 4, 4, 4, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                ]
                * 2,
            ]
        ).astype("float32")
        self.assertAllClose(outputs[POINT_CLOUDS], augmented_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], augmented_bounding_boxes)
