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

from keras_cv.models.segmentation.deeplab import SegmentationHead


class SegmentationHeadTest(tf.test.TestCase):
    def test_result_shapes(self):
        p3 = tf.ones([2, 32, 32, 3])
        p4 = tf.ones([2, 16, 16, 3])
        p5 = tf.ones([2, 8, 8, 3])
        inputs = {3: p3, 4: p4, 5: p5}

        head = SegmentationHead(classes=11)

        output = head(inputs)
        # Make sure the output shape is same as the p3
        self.assertEquals(output.shape, [2, 32, 32, 11])

    def test_invalid_input_type(self):
        p3 = tf.ones([2, 32, 32, 3])
        p4 = tf.ones([2, 16, 16, 3])
        p5 = tf.ones([2, 8, 8, 3])
        list_input = [p3, p4, p5]

        head = SegmentationHead(classes=11)
        with self.assertRaisesRegexp(ValueError, "Expect the inputs to be a dict"):
            head(list_input)
