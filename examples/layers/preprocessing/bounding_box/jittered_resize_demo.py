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

import keras_cv


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    jittered_resize = keras_cv.layers.JitteredResize(
        target_size=(512, 512),
        scale_factor=(3 / 4, 4 / 3),
        bounding_box_format="xyxy",
        minimum_box_area_ratio=0.5
    )
    result = dataset.map(jittered_resize, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()
