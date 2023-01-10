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
import demo_utils
import tensorflow as tf
from luketils import visualization

import keras_cv


def main():
    augment = keras_cv.layers.JitteredResize(
        target_size=(640, 640), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
    )
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xywh")
    dataset = dataset.map(
        lambda x: augment(x, training=True), num_parallel_calls=tf.data.AUTOTUNE
    )
    demo_utils.visualize_data(dataset, bounding_box_format="xywh")

    dataset = dataset.map(
        lambda x: augment(x, training=False), num_parallel_calls=tf.data.AUTOTUNE
    )
    demo_utils.visualize_data(dataset, bounding_box_format="xywh")


if __name__ == "__main__":
    main()
