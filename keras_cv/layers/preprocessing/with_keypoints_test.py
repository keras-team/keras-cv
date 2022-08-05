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
    )
    def test_can_run_with_keypoints(self, layer_cls, init_args):
        layer = layer_cls(**init_args)

        img = tf.ones(shape=(3, 4, 4, 3), dtype=tf.float32)
        keypoints = tf.ones((3, 2, 6, 2), dtype=tf.float32)

        inputs = {"images": img, "keypoints": keypoints}
        outputs = layer(inputs)
        self.assertTrue("keypoints" in outputs)

    # this has to be a separate test case to exclude CutMix and MixUp
    @parameterized.named_parameters(
        *TEST_CONFIGURATIONS,
        *GEOMETRIC_TEST_CONFIGURATIONS,
    )
    def test_can_run_with_keypoints_single_image(self, layer_cls, init_args):
        layer = layer_cls(**init_args)
        img = tf.ones(shape=(4, 4, 3), dtype=tf.float32)
        keypoints = tf.ones((3, 8, 2), dtype=tf.float32)
        inputs = {"images": img, "keypoints": keypoints}
        outputs = layer(inputs)
        self.assertTrue("keypoints" in outputs)

    # CutMix needs labels data
    def test_cut_mix_keeps_keypoints_data(self):
        layer = preprocessing.CutMix()
        img = tf.ones(shape=(3, 4, 4, 3), dtype=tf.float32)
        labels = tf.ones((3), dtype=tf.float32)
        keypoints = tf.reshape(
            tf.range(3 * 2 * 6 * 2, dtype=tf.float32), shape=(3, 2, 6, 2)
        )
        inputs = {"images": img, "keypoints": keypoints, "labels": labels}
        outputs = layer(inputs)
        self.assertAllEqual(inputs["keypoints"], outputs["keypoints"])
