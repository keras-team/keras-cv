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

from keras_cv import bounding_box


class BoundingBoxUtilTestCase(tf.test.TestCase):
    def test_clip_to_image(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bboxes = tf.convert_to_tensor([[200, 200, 400, 400], [100, 100, 300, 300]])
        image = tf.ones(shape=(height, width, 3))
        bboxes_out = bounding_box.clip_to_image(
            bboxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllGreaterEqual(bboxes_out, 0)
        x1, y1, x2, y2 = tf.split(bboxes_out, 4, axis=1)
        self.assertAllLessEqual([x1, x2], width)
        self.assertAllLessEqual([y1, y2], height)
        # Test relative format batched
        image = tf.ones(shape=(1, height, width, 3))
        bboxes = tf.convert_to_tensor([[[0.2, -1, 1.2, 0.3], [0.4, 1.5, 0.2, 0.3]]])
        bboxes_out = bounding_box.clip_to_image(
            bboxes, bounding_box_format="rel_xyxy", images=image
        )
        self.assertAllGreaterEqual(bboxes_out, 0)
        self.assertAllLessEqual(bboxes_out, 1)
