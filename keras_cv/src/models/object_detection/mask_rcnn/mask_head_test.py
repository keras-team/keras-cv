# Copyright 2024 The KerasCV Authors
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

import pytest
from absl.testing import parameterized

from keras_cv.src.backend import ops
from keras_cv.src.backend.config import keras_3
from keras_cv.src.models.object_detection.mask_rcnn import MaskHead
from keras_cv.src.tests.test_case import TestCase


class RCNNHeadTest(TestCase):
    @parameterized.parameters(
        (2, 256, 20, 7, 256),
        (1, 512, 80, 14, 512),
    )
    @pytest.mark.skipif(not keras_3(), reason="disabling test for Keras 2")
    def test_mask_head_output_shapes(
        self,
        batch_size,
        num_rois,
        num_classes,
        roi_align_target_size,
        num_filters,
    ):
        layer = MaskHead(num_classes)

        inputs = ops.ones(
            shape=(
                batch_size,
                num_rois,
                roi_align_target_size,
                roi_align_target_size,
                num_filters,
            )
        )
        outputs = layer(inputs)

        mask_size = roi_align_target_size * 2

        self.assertEqual(
            (batch_size, num_rois, mask_size, mask_size, num_classes + 1),
            outputs.shape,
        )
