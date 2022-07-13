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
from absl.testing import parameterized

from keras_cv.layers import preprocessing


class RandomShearTest(tf.test.TestCase, parameterized.TestCase):
    def test_aggressive_shear_fills_at_least_some_pixels(self):
        img_shape = (50, 50, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = preprocessing.RandomShear(
            x_factor=(3, 3), seed=0, fill_mode="constant", fill_value=fill_value
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    @parameterized.parameters("x", "y")
    def test_shear_with_keypoints_dict(self, shear_type):
        image = tf.ones(shape=(100, 100, 3))
        factor = 0.5

        keypoints = np.array(
            [
                [25.0, 25.0],
                [75.0, 25.0],
                [75.0, 75.0],
                [25.0, 75.0],
            ]
        )

        expected_output = np.copy(keypoints)
        if shear_type == "x":
            shear_kwargs = {"x_factor": (factor, factor)}
            expected_output[..., 0] += keypoints[..., 1] * factor
        elif shear_type == "y":
            shear_kwargs = {"y_factor": (factor, factor)}
            expected_output[..., 1] += keypoints[..., 0] * factor

        keypoints = tf.RaggedTensor.from_row_lengths(keypoints, [2, 1, 0, 1])
        expected_output = tf.RaggedTensor.from_row_lengths(
            expected_output[[0, 1, 3], :], [2, 0, 0, 1]
        )

        shear_kwargs['keypoint_format'] = 'xy'
        layer = preprocessing.RandomShear(**shear_kwargs)
        output = layer({"images": image, "keypoints": keypoints})
        self.assertAllClose(output["keypoints"], expected_output)

    @parameterized.parameters("x", "y")
    def test_shear_with_bbox_dict(self, shear_type):
        image = tf.ones(shape=(100, 100, 3))
        factor = 0.5

        bounding_boxes = np.array([
            [25.0, 25.0, 75.0, 75.0],
        ])

        if shear_type == "x":
            shear_kwargs = {"x_factor": (factor, factor), "bounding_box_format": "xyxy"}
            expected_output = np.array([[37.5, 25.0, 100, 75.0]])
        elif shear_type == "y":
            shear_kwargs = {"y_factor": (factor, factor), "bounding_box_format": "xyxy"}
            expected_output = np.array([[25.0, 37.5, 75.0, 100]])

        layer = preprocessing.RandomShear(**shear_kwargs)
        output = layer({"images": image, "bounding_boxes": bounding_boxes})

        self.assertAllClose(output["bounding_boxes"], expected_output)
