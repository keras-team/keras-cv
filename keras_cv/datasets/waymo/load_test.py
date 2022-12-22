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
import os

import pytest
import tensorflow as tf

try:
    from keras_cv.datasets.waymo import load
except ImportError:
    # Waymo Open Dataset dependency may be missing, in which case we expect
    # these tests will be skipped based on the TEST_WAYMO_DEPS environment var.
    pass


class WaymoOpenDatasetLoadTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.test_data_path = os.path.abspath(
            os.path.join(os.path.abspath(__file__), os.path.pardir, "test_data")
        )
        self.test_data_file = "wod_one_frame.tfrecord"

    @pytest.mark.skipif(
        "TEST_WAYMO_DEPS" not in os.environ or os.environ["TEST_WAYMO_DEPS"] != "true",
        reason="Requires Waymo Open Dataset package",
    )
    def test_load_from_directory(self):
        dataset = load(self.test_data_path)

        # Extract records into a list
        dataset = [record for record in dataset]

        self.assertEquals(len(dataset), 1)
        self.assertNotEqual(dataset[0]["timestamp_micros"], 0)

    @pytest.mark.skipif(
        "TEST_WAYMO_DEPS" not in os.environ or os.environ["TEST_WAYMO_DEPS"] != "true",
        reason="Requires Waymo Open Dataset package",
    )
    def test_load_from_files(self):
        dataset = load([os.path.join(self.test_data_path, self.test_data_file)])

        # Extract records into a list
        dataset = [record for record in dataset]

        self.assertEquals(len(dataset), 1)
        self.assertNotEqual(dataset[0]["timestamp_micros"], 0)
