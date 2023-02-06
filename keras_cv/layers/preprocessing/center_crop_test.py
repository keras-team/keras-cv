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


class CenterCropTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.product(target_height=[5, 10, 15, 20], target_width=[5, 10, 15, 20])
    def test_same_results(self, target_height, target_width):
        images = tf.random.normal((2, 10, 10, 3))

        original_layer = tf.keras.layers.CenterCrop(target_height, target_width)
        layer = preprocessing.CenterCrop(target_height, target_width)

        original_output_image = original_layer(images)
        output = layer({"images": images})

        self.assertShapeEqual(original_output_image, output["images"])
        self.assertAllClose(original_output_image, output["images"])

    def test_bounding_boxes_raises(self):
        with self.assertRaises(ValueError):
            preprocessing.CenterCrop(10, 10, bounding_box_format="xyxy")

        images = tf.random.normal([2, 20, 20, 3])
        boxes = tf.constant(
            [
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
                [[0, 0, 15, 15], [0, 0, 10, 10], [0, 0, 5, 5], [18, 18, 20, 20]],
            ],
            dtype=tf.float32,
        )
        classes = tf.convert_to_tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
        bboxes = {"boxes": boxes, "classes": classes}

        layer = preprocessing.CenterCrop(10, 10)
        with self.assertRaises(ValueError):
            layer({"images": images, "bounding_boxes": bboxes})
