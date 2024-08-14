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

from absl.testing import parameterized

from keras_cv.src.backend import ops
from keras_cv.src.models.object_detection.faster_rcnn import RCNNHead
from keras_cv.src.tests.test_case import TestCase


class RCNNHeadTest(TestCase):
    @parameterized.parameters(
        (2, 512, 20, 7, 256),
        (1, 1000, 80, 14, 512),
    )
    def test_rcnn_head_output_shapes(
        self,
        batch_size,
        num_rois,
        num_classes,
        roi_align_target_size,
        num_filters,
    ):
        layer = RCNNHead(num_classes)

        feature_map_size = (roi_align_target_size**2) * num_filters
        inputs = ops.ones(shape=(batch_size, num_rois, feature_map_size))
        outputs = layer(inputs)

        self.assertEqual([batch_size, num_rois, 4], outputs[0].shape)
        self.assertEqual(
            [batch_size, num_rois, num_classes + 1], outputs[1].shape
        )
