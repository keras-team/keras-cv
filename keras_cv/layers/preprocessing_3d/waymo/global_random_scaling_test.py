# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import numpy as np
import pytest

from keras_cv.backend.config import keras_3
from keras_cv.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing_3d.waymo.global_random_scaling import (
    GlobalRandomScaling,
)
from keras_cv.tests.test_case import TestCase

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


@pytest.mark.skipif(keras_3(), reason="Not implemented in Keras 3")
class GlobalScalingTest(TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            x_factor=(0.5, 1.5),
            y_factor=(0.5, 1.5),
            z_factor=(0.5, 1.5),
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_augment_point_clouds_and_bounding_boxes_with_same_scaling(self):
        add_layer = GlobalRandomScaling(
            x_factor=(0.5, 1.5),
            y_factor=(0.5, 1.5),
            z_factor=(0.5, 1.5),
            preserve_aspect_ratio=True,
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            x_factor=(1.0, 1.0),
            y_factor=(1.0, 1.0),
            z_factor=(1.0, 1.0),
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_2x_scaling_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            x_factor=(2.0, 2.0),
            y_factor=(2.0, 2.0),
            z_factor=(2.0, 2.0),
        )
        point_clouds = np.array(
            [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2
        ).astype("float32")
        bounding_boxes = np.array([[[0, 1, 2, 3, 4, 5, 6]] * 2] * 2).astype(
            "float32"
        )
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        scaled_point_clouds = np.array(
            [[[0, 2, 4, 3, 4, 5, 6, 7, 8, 9]] * 2] * 2
        ).astype("float32")
        scaled_bounding_boxes = np.array(
            [[[0, 2, 4, 6, 8, 10, 6]] * 2] * 2
        ).astype("float32")
        self.assertAllClose(outputs[POINT_CLOUDS], scaled_point_clouds)
        self.assertAllClose(outputs[BOUNDING_BOXES], scaled_bounding_boxes)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            x_factor=(0.5, 1.5),
            y_factor=(0.5, 1.5),
            z_factor=(0.5, 1.5),
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomScaling(
            x_factor=(1.0, 1.0),
            y_factor=(1.0, 1.0),
            z_factor=(1.0, 1.0),
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)
