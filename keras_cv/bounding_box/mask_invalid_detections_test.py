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

import numpy as np
import pytest
import tensorflow as tf

from keras_cv import bounding_box
from keras_cv.backend import ops
from keras_cv.tests.test_case import TestCase


class MaskInvalidDetectionsTest(TestCase):
    def test_correctly_masks_based_on_max_dets(self):
        bounding_boxes = {
            "boxes": ops.random.uniform((4, 100, 4)),
            "num_detections": ops.array([2, 3, 4, 2]),
            "classes": ops.random.uniform((4, 100)),
        }

        result = bounding_box.mask_invalid_detections(bounding_boxes)

        negative_one_boxes = result["boxes"][:, 5:, :]
        self.assertAllClose(
            negative_one_boxes, -np.ones_like(negative_one_boxes)
        )

        preserved_boxes = result["boxes"][:, :2, :]
        self.assertAllClose(preserved_boxes, bounding_boxes["boxes"][:, :2, :])

        boxes_from_image_3 = result["boxes"][2, :4, :]
        self.assertAllClose(
            boxes_from_image_3, bounding_boxes["boxes"][2, :4, :]
        )

    @pytest.mark.tf_keras_only
    def test_ragged_outputs(self):
        bounding_boxes = {
            "boxes": np.stack(
                [
                    np.random.uniform(size=(10, 4)),
                    np.random.uniform(size=(10, 4)),
                ]
            ),
            "num_detections": np.array([2, 3]),
            "classes": np.stack(
                [np.random.uniform(size=(10,)), np.random.uniform(size=(10,))]
            ),
        }

        result = bounding_box.mask_invalid_detections(
            bounding_boxes, output_ragged=True
        )
        self.assertTrue(isinstance(result["boxes"], tf.RaggedTensor))
        self.assertEqual(result["boxes"][0].shape[0], 2)
        self.assertEqual(result["boxes"][1].shape[0], 3)

    @pytest.mark.tf_keras_only
    def test_correctly_masks_confidence(self):
        bounding_boxes = {
            "boxes": np.stack(
                [
                    np.random.uniform(size=(10, 4)),
                    np.random.uniform(size=(10, 4)),
                ]
            ),
            "confidence": np.random.uniform(size=(2, 10)),
            "num_detections": np.array([2, 3]),
            "classes": np.stack(
                [np.random.uniform(size=(10,)), np.random.uniform(size=(10,))]
            ),
        }

        result = bounding_box.mask_invalid_detections(
            bounding_boxes, output_ragged=True
        )
        self.assertTrue(isinstance(result["boxes"], tf.RaggedTensor))
        self.assertEqual(result["boxes"][0].shape[0], 2)
        self.assertEqual(result["boxes"][1].shape[0], 3)
        self.assertEqual(result["confidence"][0].shape[0], 2)
        self.assertEqual(result["confidence"][1].shape[0], 3)
