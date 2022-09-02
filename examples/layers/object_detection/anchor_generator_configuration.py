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

from keras_cv import layers as cv_layers


def _default_anchor_generator(bounding_box_format):
    strides = [50]
    sizes = [100.0]
    scales = [1.0]
    aspect_ratios = [1.0]
    return cv_layers.AnchorGenerator(
        bounding_box_format=bounding_box_format,
        anchor_sizes=sizes,
        aspect_ratios=aspect_ratios,
        scales=scales,
        strides=strides,
        clip_boxes=True,
    )


generator = _default_anchor_generator(bounding_box_format="xywh")


def pair_with_anchor_boxes(inputs):
    images = inputs["images"]
    anchor_boxes = generator(images[0])
    anchor_boxes = anchor_boxes[0]
    anchor_boxes = tf.expand_dims(anchor_boxes, axis=0)
    anchor_boxes = tf.tile(anchor_boxes, [tf.shape(images)[0], 1, 1])
    inputs["bounding_boxes"] = anchor_boxes
    return inputs


if __name__ == "__main__":
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xywh")
    result = dataset.map(pair_with_anchor_boxes, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(result, bounding_box_format="xywh")
