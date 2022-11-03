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
import tensorflow_datasets as tfds

import keras_cv
from keras_cv import layers


def preproc(inputs):
    image = inputs["image"]
    image = tf.cast(image, tf.float32)
    gt_boxes = inputs["objects"]["bbox"]
    gt_boxes = keras_cv.bounding_box.convert_format(
        gt_boxes,
        images=image,
        source="rel_yxyx",
        target="yxyx",
    )
    gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
    gt_classes = tf.expand_dims(gt_classes, axis=-1)
    bounding_boxes = tf.concat([gt_boxes, gt_classes], axis=-1)
    bounding_boxes = keras_cv.bounding_box.convert_format(
        bounding_boxes, images=image, source="yxyx", target="xywh"
    )
    return {"images": image, "bounding_boxes": bounding_boxes}


# pad to aspect ratio height > width
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
)
train_ds = train_ds.map(preproc)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(16, drop_remainder=True)
)
resizing = layers.Resizing(
    height=600, width=400, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)
train_ds = train_ds.map(resizing)
demo_utils.visualize_data(train_ds, bounding_box_format="xywh")

# pad to aspect ratio width > height
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
)
train_ds = train_ds.map(preproc)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(16, drop_remainder=True)
)
resizing = layers.Resizing(
    height=400, width=600, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)
train_ds = train_ds.map(resizing)
demo_utils.visualize_data(train_ds, bounding_box_format="xywh")
