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

from keras_cv.tests.test_case import TestCase
from keras_cv.utils import fill_utils


class BoundingBoxToMaskTest(TestCase):
    def _run_test(self, corners, expected):
        mask = fill_utils.corners_to_mask(corners, mask_shape=(6, 6))
        mask = tf.cast(mask, dtype="int32")
        tf.assert_equal(mask, expected)

    def test_corners_whole(self):
        expected = np.array(
            [
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype="int32",
        )
        corners = np.array([[1, 0, 4, 3]], dtype="float32")
        self._run_test(corners, expected)

    def test_corners_frac(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[1.5, 0.5, 4.5, 3.5]], dtype="float32")
        self._run_test(corners, expected)

    def test_width_zero(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[0, 0, 0, 3]], dtype="float32")
        self._run_test(corners, expected)

    def test_height_zero(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[1, 0, 4, 0]], dtype="float32")
        self._run_test(corners, expected)

    def test_width_negative(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[1, 0, -2, 3]], dtype="float32")
        self._run_test(corners, expected)

    def test_height_negative(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[1, 0, 4, -2]], dtype="float32")
        self._run_test(corners, expected)

    def test_width_out_of_lower_bound(self):
        expected = np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[-2, -2, 2, 3]], dtype="float32")
        self._run_test(corners, expected)

    def test_width_out_of_upper_bound(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[4, 0, 8, 3]], dtype="float32")
        self._run_test(corners, expected)

    def test_height_out_of_lower_bound(self):
        expected = np.array(
            [
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[1, -3, 4, 2]], dtype="float32")
        self._run_test(corners, expected)

    def test_height_out_of_upper_bound(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 0, 0],
            ]
        )
        corners = np.array([[1, 4, 4, 9]], dtype="float32")
        self._run_test(corners, expected)

    def test_start_out_of_upper_bound(self):
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        corners = np.array([[8, 8, 10, 12]], dtype="float32")
        self._run_test(corners, expected)


class FillRectangleTest(TestCase):
    def _run_test(self, img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected):
        batch_size = 1

        batch_shape = (batch_size, img_h, img_w, 1)
        images = np.ones(batch_shape, dtype="int32")

        centers_x = tf.fill([batch_size], cent_x)
        centers_y = tf.fill([batch_size], cent_y)
        width = tf.fill([batch_size], rec_w)
        height = tf.fill([batch_size], rec_h)
        fill = tf.zeros_like(images)

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove batch dimension and channel dimension
        filled_images = filled_images[0, ..., 0]

        tf.assert_equal(filled_images, expected)

    def test_rectangle_position(self):
        img_w, img_h = 8, 8
        cent_x, cent_y = 4, 3
        rec_w, rec_h = 5, 3
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype="int32",
        )
        self._run_test(img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected)

    def test_width_out_of_lower_bound(self):
        img_w, img_h = 8, 8
        cent_x, cent_y = 1, 3
        rec_w, rec_h = 5, 3
        # assert width is truncated when cent_x - rec_w < 0
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype="int32",
        )
        self._run_test(img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected)

    def test_width_out_of_upper_bound(self):
        img_w, img_h = 8, 8
        cent_x, cent_y = 6, 3
        rec_w, rec_h = 5, 3
        # assert width is truncated when cent_x + rec_w > img_w
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype="int32",
        )
        self._run_test(img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected)

    def test_height_out_of_lower_bound(self):
        img_w, img_h = 8, 8
        cent_x, cent_y = 4, 1
        rec_w, rec_h = 3, 5
        # assert height is truncated when cent_y - rec_h < 0
        expected = np.array(
            [
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype="int32",
        )
        self._run_test(img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected)

    def test_height_out_of_upper_bound(self):
        img_w, img_h = 8, 8
        cent_x, cent_y = 4, 6
        rec_w, rec_h = 3, 5
        # assert height is truncated when cent_y + rec_h > img_h
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 0, 1, 1],
            ],
            dtype="int32",
        )
        self._run_test(img_w, img_h, cent_x, cent_y, rec_w, rec_h, expected)

    def test_different_fill(self):
        batch_size = 2
        img_w, img_h = 5, 5
        cent_x, cent_y = 2, 2
        rec_w, rec_h = 3, 3

        batch_shape = (batch_size, img_h, img_w, 1)
        images = np.ones(batch_shape, dtype="int32")

        centers_x = tf.fill([batch_size], cent_x)
        centers_y = tf.fill([batch_size], cent_y)
        width = tf.fill([batch_size], rec_w)
        height = tf.fill([batch_size], rec_h)
        fill = tf.stack(
            [tf.fill(images[0].shape, 2), tf.fill(images[1].shape, 3)]
        )

        filled_images = fill_utils.fill_rectangle(
            images, centers_x, centers_y, width, height, fill
        )
        # remove channel dimension
        filled_images = filled_images[..., 0]

        expected = np.array(
            [
                [
                    [1, 1, 1, 1, 1],
                    [1, 2, 2, 2, 1],
                    [1, 2, 2, 2, 1],
                    [1, 2, 2, 2, 1],
                    [1, 1, 1, 1, 1],
                ],
                [
                    [1, 1, 1, 1, 1],
                    [1, 3, 3, 3, 1],
                    [1, 3, 3, 3, 1],
                    [1, 3, 3, 3, 1],
                    [1, 1, 1, 1, 1],
                ],
            ],
            dtype="int32",
        )
        tf.assert_equal(filled_images, expected)
