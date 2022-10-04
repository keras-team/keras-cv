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

from keras_cv.models.segmentation.__internal__ import SegmentationHead


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

    def test_scale_up_output(self):
        p3 = tf.ones([2, 32, 32, 3])
        p4 = tf.ones([2, 16, 16, 3])
        p5 = tf.ones([2, 8, 8, 3])
        inputs = {3: p3, 4: p4, 5: p5}

        head = SegmentationHead(classes=11, output_scale_factor=4)

        output = head(inputs)
        # The output shape will scale up 4x
        self.assertEquals(output.shape, [2, 32 * 4, 32 * 4, 11])

    def test_dtype_for_classification_head(self):
        p3 = tf.ones([2, 32, 32, 3])
        p4 = tf.ones([2, 16, 16, 3])
        p5 = tf.ones([2, 8, 8, 3])
        inputs = {3: p3, 4: p4, 5: p5}
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            head = SegmentationHead(classes=11, output_scale_factor=4)

            _ = head(inputs)

            # Make sure the dtype of the classification head is still float32, which will
            # avoid NAN loss issue for mixed precision
            self.assertEquals(head._classification_layer.dtype, tf.float32)
            self.assertEquals(head._classification_layer.compute_dtype, tf.float32)
        finally:
            tf.keras.mixed_precision.set_global_policy(None)
