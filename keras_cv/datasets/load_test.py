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
"""Tests ability to load keras_cv.datasets."""

from keras_cv.tests.test_case import TestCase


class LoadTest(TestCase):
    def test_datasets_load(self):
        import keras_cv.datasets  # noqa: F401

    def test_imagenet_load(self):
        import keras_cv.datasets.imagenet  # noqa: F401
