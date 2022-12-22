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

from keras_cv.models.__internal__.unet import UNet


class UNetTest(tf.test.TestCase):
    # This test is disabled because it requires tf-nightly to run
    # (tf-nightly includes the synchronized param for BatchNorm layer)
    def disabled_test_example_unet_output_shape(self):
        x = tf.random.normal((1, 16, 16, 5))
        output = UNet([(128, 6), (256, 2), (512, 1)], [512, 256, 256])(x)
        self.assertEqual(output.shape, x.shape[:-1] + (256))
