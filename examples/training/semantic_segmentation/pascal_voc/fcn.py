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

import sys

import tensorflow as tf
from absl import flags
from absl import logging

import keras_cv
import keras_cv.layers.preprocessing as preprocessing
from keras_cv.datasets.pascal_voc.segmentation import load
from keras_cv.models.segmentation.fcn import FCN8S, FCN16S, FCN32S

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

if FLAGS.mixed_precision:
    logging.info("mixed precision training enabled")
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
print("Number of accelerators: ", strategy.num_replicas_in_sync)

# parameters from FasterRCNN [paper](https://arxiv.org/pdf/1506.01497.pdf)

local_batch = 20
global_batch = local_batch * strategy.num_replicas_in_sync
base_lr = 0.007 * global_batch / 16

# TODO(tanzhenyu): add a diff dataset.
# all_ds = load(split="sbd_train", data_dir=None)
# all_ds = all_ds.concatenate(load(split="sbd_eval", data_dir=None))
# train_ds = all_ds.take(10000)
# eval_ds = all_ds.skip(10000).concatenate(load(split="diff", data_dir=None))
train_ds = load(split="sbd_train", data_dir=None)
eval_ds = load(split="sbd_eval", data_dir=None)


def resize_image(img, cls_seg, augment=False):
    img = tf.keras.layers.Resizing(224, 224, interpolation="nearest")(img)
    cls_seg = tf.keras.layers.Resizing(224, 224, interpolation="nearest")(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.int32)

    inputs = {"images": img, "segmentation_masks": cls_seg}
    return inputs["images"], inputs["segmentation_masks"]


def process(examples):
    image = examples.pop("image")
    cls_seg = examples.pop("class_segmentation")

    image, cls_seg = resize_image(image, cls_seg)

    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg

train_ds = train_ds.map(lambda x: process(x, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(16, drop_remainder=True)

eval_ds = eval_ds.map(lambda x: process(x), num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(16, drop_remainder=True)

train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    backbone = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    model = FCN8S(classes=21, backbone=backbone)    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

model.fit(train_ds, epochs=100, validation_data=eval_ds, callbacks=callbacks)
