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
mosaic_demo.py shows how to use the Mosaic preprocessing layer for
object detection.
"""
import demo_utils
import tensorflow as tf

from keras_cv import layers
import keras_cv

def main():
    dataset, _ = keras_cv.datasets.pascal_voc.load(
        split='train',
        bounding_box_format="xyxy",
        batch_size=9,
        img_size=(512, 512)
    )
    # ragged tensor of images
    # sample = next(iter(dataset))
    # resizing = layers.Resizing(height=512, width=512, bounding_box_format='xyxy')
    # outputs = resizing(sample)
    # result = dataset.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(dataset, bounding_box_format="xyxy")


if __name__ == "__main__":
    main()
