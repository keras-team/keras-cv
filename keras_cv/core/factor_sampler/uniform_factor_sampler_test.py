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

import keras_cv


class UniformFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_cv.UniformFactorSampler(0.3, 0.6)
        self.assertTrue(0.3 <= factor() <= 0.6)

    def test_config(self):
        factor = keras_cv.UniformFactorSampler(0.3, 0.6)
        config = factor.get_config()
        self.assertEqual(config["lower"], 0.3)
        self.assertEqual(config["upper"], 0.6)
