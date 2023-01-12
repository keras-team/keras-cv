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
"""
resizing_demo.py shows how to use the resizing preprocessing layer for
object detection.
"""
import demo_utils
import tensorflow as tf
import tensorflow_datasets as tfds

import keras_cv
from keras_cv import layers


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xywh")
    resizing = layers.Resizing(
        height=300, width=400, pad_to_aspect_ratio=True, bounding_box_format="xywh"
    )
    dataset = dataset.map(resizing)
    demo_utils.visualize_data(dataset, bounding_box_format="xywh")


if __name__ == "__main__":
    main()
