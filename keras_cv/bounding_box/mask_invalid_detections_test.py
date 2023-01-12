# Copyright 2023 The KerasCV Authors
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


class MaskInvalidDetectionsTest(tf.test.TestCase):
    def test_correctly_masks_based_on_max_dets(self):
        bounding_boxes = {
            "boxes": tf.random.uniform((4, 100, 4)),
            "num_detections": tf.constant([2, 3, 4, 1]),
            "classes": tf.random.uniform((4, 100)),
        }

        result = bounding_box.mask_invalid_detections(bounding_boxes)

        negative_one_boxes = result["boxes"][:, 5:, :]
        self.assertAllClose(negative_one_boxes, -tf.ones_like(negative_one_boxes))

    def test_preserves_ragged(self):
        bounding_boxes = {
            "boxes": tf.ragged.stack(
                [tf.random.uniform(5, 4), tf.random.uniform(10, 4)]
            ),
            "num_detections": tf.constant([2, 3]),
            "classes": tf.ragged.stack([tf.random.uniform(5), tf.random.uniform(5)]),
        }

        result = bounding_box.mask_invalid_detections(bounding_boxes)
        self.assertTrue(isinstance(result, tf.RaggedTensor))
