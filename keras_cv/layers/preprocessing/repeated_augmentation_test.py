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
import pytest
import tensorflow as tf

import keras_cv.layers as cv_layers


class RepeatedAugmentationTest(tf.test.TestCase):
    def test_output_shapes(self):
        repeated_augment = cv_layers.RepeatedAugmentation(
            augmenters=[
                cv_layers.RandAugment(value_range=(0, 255)),
                cv_layers.RandomFlip(),
            ]
        )
        inputs = {
            "images": tf.ones((8, 512, 512, 3)),
            "labels": tf.ones((8,)),
        }
        outputs = repeated_augment(inputs)

        self.assertEqual(outputs["images"].shape, (16, 512, 512, 3))
        self.assertEqual(outputs["labels"].shape, (16,))
