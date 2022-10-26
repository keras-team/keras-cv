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
Title: Train an Semantic Segmentation Model on Pascal VOC 2012 using KerasCV
Author: [tanzhenyu](https://github.com/tanzhenyu)
Date created: 2022/10/25
Last modified: 2022/10/25
Description: Use KerasCV to train a DeepLabV3 on Pascal VOC 2012.
"""

import sys

import tensorflow as tf
from absl import flags

import keras_cv
from keras_cv.datasets.pascal_voc.segmentation import load
from keras_cv.models.segmentation.deeplab import DeepLabV3

flags.DEFINE_string(
    "weights_path",
    "weights_{epoch:02d}.h5",
    "Directory which will be used to store weight checkpoints.",
)
flags.DEFINE_string(
    "tensorboard_path",
    "logs",
    "Directory which will be used to store tensorboard logs.",
)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
print("Number of accelerators: ", strategy.num_replicas_in_sync)

# parameters from FasterRCNN [paper](https://arxiv.org/pdf/1506.01497.pdf)

local_batch = 4
global_batch = local_batch * strategy.num_replicas_in_sync
base_lr = 0.007 * global_batch / 16

train_ds = load(split="train", data_dir=None)
eval_ds = load(split="eval", data_dir=None)

resize_layer = tf.keras.layers.Resizing(512, 512)


def proc_train_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    image = resize_layer(image)
    cls_seg = examples.pop("class_segmentation")
    zeros = tf.zeros_like(cls_seg)
    mask = tf.equal(cls_seg, 255)
    cls_seg = tf.where(mask, zeros, cls_seg)
    cls_seg = resize_layer(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.uint32)
    return image, cls_seg


train_ds = train_ds.map(proc_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(global_batch)

eval_ds = eval_ds.map(proc_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(global_batch)

train_ds = train_ds.shuffle(8)

with strategy.scope():
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[30000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr],
    )
    backbone = keras_cv.models.ResNet50V2(
        include_rescaling=True, weights="imagenet", include_top=False
    ).as_backbone()
    model = DeepLabV3(classes=21, backbone=backbone, include_rescaling=True)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, clipnorm=10.0
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
    ]

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.weights_path,
        monitor="val_mean_io_u",
        save_best_only=True,
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.tensorboard_path, write_steps_per_second=True
    ),
]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

model.fit(train_ds, epochs=100, validation_data=eval_ds, callbacks=callbacks)
