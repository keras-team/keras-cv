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
import pathlib
import sys

import tensorflow as tf
from absl import flags

from keras_cv.datasets.waymo import load


def simple_transformer(frame):
    return {"timestamp_micros": frame.timestamp_micros}


class WaymoOpenDatasetLoadTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.test_data_path = os.path.abspath(
            os.path.join(os.path.abspath(__file__), os.path.pardir, "test_data")
        )
        self.test_data_file = "mini.tfrecord"
        self.output_signature = {"timestamp_micros": tf.TensorSpec((), tf.int64)}

    def test_load_from_directory(self):
        dataset = load(self.test_data_path, simple_transformer, self.output_signature)

        # Extract records into a list
        dataset = [record for record in dataset]

        self.assertEquals(len(dataset), 1)
        self.assertEquals(dataset[0]["timestamp_micros"], 8675309)

    def test_load_from_files(self):
        dataset = load(
            [os.path.join(self.test_data_path, self.test_data_file)],
            simple_transformer,
            self.output_signature,
        )

        # Extract records into a list
        dataset = [record for record in dataset]

        self.assertEquals(len(dataset), 1)
        self.assertEquals(dataset[0]["timestamp_micros"], 8675309)
