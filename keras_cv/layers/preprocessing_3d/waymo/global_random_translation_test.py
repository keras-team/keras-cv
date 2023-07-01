# Copyright 2022 Waymo LLC.
#
# Licensed under the terms in https://github.com/keras-team/keras-cv/blob/master/keras_cv/layers/preprocessing_3d/waymo/LICENSE  # noqa: E501

import numpy as np
import tensorflow as tf

from keras_cv.layers.preprocessing_3d import base_augmentation_layer_3d
from keras_cv.layers.preprocessing_3d.waymo.global_random_translation import (
    GlobalRandomTranslation,
)

POINT_CLOUDS = base_augmentation_layer_3d.POINT_CLOUDS
BOUNDING_BOXES = base_augmentation_layer_3d.BOUNDING_BOXES


class GlobalRandomTranslationTest(tf.test.TestCase):
    def test_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomTranslation(
            x_stddev=1.0, y_stddev=1.0, z_stddev=1.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomTranslation(
            x_stddev=0.0, y_stddev=0.0, z_stddev=0.0
        )
        point_clouds = np.random.random(size=(2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)

    def test_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomTranslation(
            x_stddev=1.0, y_stddev=1.0, z_stddev=1.0
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertNotAllClose(inputs, outputs)

    def test_not_augment_batch_point_clouds_and_bounding_boxes(self):
        add_layer = GlobalRandomTranslation(
            x_stddev=0.0, y_stddev=0.0, z_stddev=0.0
        )
        point_clouds = np.random.random(size=(3, 2, 50, 10)).astype("float32")
        bounding_boxes = np.random.random(size=(3, 2, 10, 7)).astype("float32")
        inputs = {POINT_CLOUDS: point_clouds, BOUNDING_BOXES: bounding_boxes}
        outputs = add_layer(inputs)
        self.assertAllClose(inputs, outputs)
