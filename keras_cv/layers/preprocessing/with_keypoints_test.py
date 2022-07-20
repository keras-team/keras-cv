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
from absl.testing import parameterized

from keras_cv.layers import preprocessing

# Imports from with_labels the list of augmentation that should perform a No-Op.
from keras_cv.layers.preprocessing.with_labels_test import TEST_CONFIGURATIONS

# List here the layers that are expected to modify keypoints with
# their specific parameters.
GEOMETRIC_TEST_CONFIGURATIONS = []


class WithKeypointsTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        *GEOMETRIC_TEST_CONFIGURATIONS,
        ("CutMix", preprocessing.CutMix, {}),
    )
    def test_can_run_with_keypoints(self, layer_cls, init_args):
        if "clip_points_to_image_size" in init_args:
            init_args["clip_points_to_image_size"] = False
        layer = layer_cls(**init_args)

        img = tf.random.uniform(
            shape=(3, 512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        labels = tf.ones((3, 2), dtype=tf.float32)
        keypoints = tf.ones((3, 2, 6, 2), dtype=tf.float32)

        inputs = {"images": img, "labels": labels, "keypoints": keypoints}
        outputs = layer(inputs)
        self.assertTrue("keypoints" in outputs)

    # this has to be a separate test case to exclude CutMix and MixUp
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        *GEOMETRIC_TEST_CONFIGURATIONS,
    )
    def test_can_run_with_keypoints_single_image(self, layer_cls, init_args):
        if "clip_points_to_image_size" in init_args:
            init_args["clip_points_to_image_size"] = True
        layer = layer_cls(**init_args)
        img = tf.random.uniform(
            shape=(512, 512, 3), minval=0, maxval=1, dtype=tf.float32
        )
        labels = tf.ones((3), dtype=tf.float32)
        keypoints = tf.ones((3, 8, 2), dtype=tf.float32)
        inputs = {"images": img, "labels": labels, "keypoints": keypoints}
        outputs = layer(inputs)
        self.assertTrue("keypoints" in outputs)
