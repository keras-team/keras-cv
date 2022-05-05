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

from keras_cv import core


class NormalFactorTest(tf.test.TestCase):
    def test_sample(self):
        factor = core.NormalFactor(mean=0.5, stddev=0.2, min_value=0, max_value=1)
        self.assertTrue(0 <= factor() <= 1)

    def test_config(self):
        factor = core.NormalFactor(mean=0.5, stddev=0.2, min_value=0, max_value=1)
        config = factor.get_config()
        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)
