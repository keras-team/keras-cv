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

from keras_cv import layers
from keras_cv.backend import ops
from keras_cv.tests.test_case import TestCase


class AugmenterTest(TestCase):
    def test_call(self):
        images = ops.ones((2, 256, 256, 3))
        augmenter = layers.Augmenter(
            [
                layers.RandomFlip(),
                layers.RandAugment(value_range=(0, 255)),
            ]
        )
        output = augmenter(images)
        self.assertEquals(output.shape, images.shape)

    def test_call_with_labels(self):
        images = {
            "labels": ops.ones((2,)),
            "images": ops.ones((2, 256, 256, 3)),
        }
        augmenter = layers.Augmenter(
            [
                layers.RandomFlip(),
                layers.RandAugment(value_range=(0, 255)),
                layers.CutMix(),
            ]
        )
        output = augmenter(images)
        self.assertEquals(output["images"].shape, images["images"].shape)
