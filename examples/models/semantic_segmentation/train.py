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
Title: Train an Object Detection Model on Pascal VOC 2007 using KerasCV
Author: [tanzhenyu](https://github.com/tanzhenyu)
Date created: 2022/09/27
Last modified: 2022/09/27
Description: Use KerasCV to train a RetinaNet on Pascal VOC 2007.
"""

import tensorflow as tf

import keras_cv
from keras_cv.datasets.pascal_voc.segmentation import load_sbd

from keras_cv.models.segmentation.deeplab import DeepLabV3


# train_ds = load_sbd("train", data_dir="/home/overflow/.keras/datasets/")
# for examples in train_ds.take(100):
#     print(examples["class_segmentation"].shape)

strategy = tf.distribute.MirroredStrategy()
local_batch = 2
global_batch = local_batch * strategy.num_replicas_in_sync

train_ds = load_sbd(split="train", data_dir="/home/overflow/.keras/datasets/")
eval_ds = load_sbd(split="eval", data_dir="/home/overflow/.keras/datasets/")
resize_layer = tf.keras.layers.Resizing(512, 512)

def proc_train_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    image = resize_layer(image)
    cls_seg = examples.pop("class_segmentation")
    cls_seg = cls_seg[..., tf.newaxis]
    zeros = tf.zeros_like(cls_seg)
    mask = tf.equal(cls_seg, 255)
    cls_seg = tf.where(mask, zeros, cls_seg)
    cls_seg = resize_layer(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.uint32)
    return image, cls_seg

train_ds = train_ds.map(proc_train_fn, num_parallel_calls=1)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)

eval_ds = eval_ds.map(proc_train_fn, num_parallel_calls=1)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)

def pad_fn(image, y_true):
    if isinstance(image, tf.RaggedTensor):
        image = image.to_tensor()
    if isinstance(y_true, tf.RaggedTensor):
        y_true = y_true.to_tensor()
    return image, y_true


train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)

with strategy.scope():
    backbone = keras_cv.models.ResNet50V2(
        include_rescaling=True, weights="imagenet/classification-v2", include_top=False
    ).as_backbone()
    model = DeepLabV3(classes=21, backbone=backbone, include_rescaling=True)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.0001, momentum=0.9, clipnorm=2.0
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(),
        tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
    ]
model.compile(optimizer, loss="SparseCategoricalCrossentropy", metrics=metrics)
model.fit(train_ds, epochs=50, validation_data=eval_ds)