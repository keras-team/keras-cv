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
Title: Train a Semantic Segmentation Model on Pascal VOC 2012 using KerasCV
Author: [tanzhenyu](https://github.com/tanzhenyu)
Date created: 2022/10/25
Last modified: 2022/10/25
Description: Use KerasCV to train a DeepLabV3 on Pascal VOC 2012.
"""

import sys

import tensorflow as tf
from absl import flags
from absl import logging

from keras_cv import models
from keras_cv.datasets.pascal_voc.segmentation import load

flags.DEFINE_string(
    "weights_path",
    "weights_{epoch:02d}.h5",
    "Directory which will be used to store weight checkpoints.",
)
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether or not to use FP16 mixed precision for training.",
)
flags.DEFINE_string(
    "tensorboard_path",
    "logs",
    "Directory which will be used to store tensorboard logs.",
)
flags.DEFINE_integer(
    "epochs",
    100,
    "Number of epochs to run for.",
)

flags.DEFINE_string(
    "model_name",
    None,
    "The model name to be trained",
)

flags.DEFINE_string(
    "model_kwargs",
    "{}",
    "Keyword argument dictionary to pass to the constructor of the model being"
    " trained",
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.mixed_precision:
    logging.info("mixed precision training enabled")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Try to detect an available TPU. If none is present, default to
# MirroredStrategy
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

# TODO(tanzhenyu): add a diff dataset.
# all_ds = load(split="sbd_train", data_dir=None)
# all_ds = all_ds.concatenate(load(split="sbd_eval", data_dir=None))
# train_ds = all_ds.take(10000)
# eval_ds = all_ds.skip(10000).concatenate(load(split="diff", data_dir=None))
train_ds = load(split="sbd_train", data_dir=None)
eval_ds = load(split="sbd_eval", data_dir=None)

resize_layer = tf.keras.layers.Resizing(512, 512, interpolation="nearest")

image_size = [512, 512, 3]


# TODO(tanzhenyu): move to KPL.
def flip_fn(image, cls_seg):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        cls_seg = tf.reverse(cls_seg, axis=[1])
    return image, cls_seg


def proc_train_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    image = resize_layer(image)
    cls_seg = examples.pop("class_segmentation")
    cls_seg = tf.cast(cls_seg, tf.float32)
    cls_seg = resize_layer(cls_seg)
    image, cls_seg = flip_fn(image, cls_seg)
    cls_seg = tf.cast(cls_seg, tf.uint8)
    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg


def proc_eval_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    image = resize_layer(image)
    cls_seg = examples.pop("class_segmentation")
    cls_seg = tf.cast(cls_seg, tf.float32)
    cls_seg = resize_layer(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.uint8)
    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg


train_ds = train_ds.map(proc_train_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(global_batch, drop_remainder=True)

eval_ds = eval_ds.map(proc_eval_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(global_batch, drop_remainder=True)

train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(2)

with strategy.scope():
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[30000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr],
    )
    backbone = models.ResNet50V2(
        include_rescaling=True,
        # This argument gives a 2% mIoU increase
        stackwise_dilations=[1, 1, 1, 2],
        input_shape=(512, 512, 3),
        include_top=False,
        weights="imagenet",
    )
    model = models.__dict__[FLAGS.model_name]
    model = model(num_classes=21, backbone=backbone, **eval(FLAGS.model_kwargs))
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, clipnorm=10.0
    )
    # ignore 255 as the class for semantic boundary.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=255)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(ignore_class=255),
        tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
        tf.keras.metrics.SparseCategoricalAccuracy(),
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

model.fit(
    train_ds, epochs=FLAGS.epochs, validation_data=eval_ds, callbacks=callbacks
)
