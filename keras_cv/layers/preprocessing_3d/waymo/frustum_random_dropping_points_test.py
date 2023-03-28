# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/LICENSE
# and https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/wdl_limited/camera/PATENTS

import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing_3d.frustum_random_dropping_points import (
    FrustumRandomDroppingPoints,
)

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class FrustumRandomDroppingPointTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0, theta_width=1, phi_width=1, drop_rate=0.5
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_drop_rate0_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0, theta_width=1, phi_width=1, drop_rate=0.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_not_augment_drop_rate1_frustum_empty_point_clouds_and_bounding_boxes(
        self,
    ):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=10, theta_width=0, phi_width=0, drop_rate=1.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_drop_rate1_large_frustum_drop_all_point_clouds(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0, theta_width=np.pi, phi_width=np.pi, drop_rate=1.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs[POINT_CLOUDS] * 0.0, outputs[POINT_CLOUDS])

    def test_exclude_all_points(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0,
            theta_width=np.pi,
            phi_width=np.pi,
            drop_rate=1.0,
            exclude_classes=1,
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        exclude_classes = np.ones(shape=(2, 50, 1)).astype("float32")
        point_clouds = np.concatenate([point_clouds, exclude_classes], axis=-1)

        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_exclude_the_first_half_points(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0,
            theta_width=np.pi,
            phi_width=np.pi,
            drop_rate=1.0,
            exclude_classes=[1, 2],
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        class_1 = np.ones(shape=(2, 10, 1)).astype("float32")
        class_2 = np.ones(shape=(2, 15, 1)).astype("float32") * 2
        classes = np.concatenate(
            [class_1, class_2, np.zeros(shape=(2, 25, 1)).astype("float32")],
            axis=1,
        )
        point_clouds = np.concatenate([point_clouds, classes], axis=-1)

        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(
            inputs[POINT_CLOUDS][:, 25:, :] * 0.0,
            outputs[POINT_CLOUDS][:, 25:, :],
        )
        self.assertAllClose(
            inputs[POINT_CLOUDS][:, :25, :], outputs[POINT_CLOUDS][:, :25, :]
        )

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = FrustumRandomDroppingPoints(
            r_distance=0, theta_width=1, phi_width=1, drop_rate=0.5
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)
