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
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags

import keras_cv
from keras_cv.callbacks import PyCOCOCallback

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

# parameters from FasterRCNN [paper](https://arxiv.org/pdf/1506.01497.pdf)

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
print("Number of accelerators: ", strategy.num_replicas_in_sync)

local_batch = 4
global_batch = local_batch * strategy.num_replicas_in_sync
base_lr = 0.01 * global_batch / 16

IMG_SIZE = 640
image_size = [IMG_SIZE, IMG_SIZE, 3]
train_ds = tfds.load(
    "voc/2007", split="train+validation", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)


def unpackage_inputs(bounding_box_format):
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        gt_boxes = tf.cast(inputs["objects"]["bbox"], tf.float32)
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        return {
            "images": image,
            "bounding_boxes": {"boxes": gt_boxes, "classes": gt_classes},
        }

    return apply


def dict_to_tuple(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_inputs("xywh"), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(8 * strategy.num_replicas_in_sync)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)


eval_ds = eval_ds.map(
    unpackage_inputs("xywh"),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)


"""
Our data pipeline is now complete.  We can now move on to data augmentation:
"""
eval_resizing = keras_cv.layers.Resizing(
    IMG_SIZE, IMG_SIZE, bounding_box_format="yxyx", pad_to_aspect_ratio=True
)

augmenter = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="yxyx"),
        keras_cv.layers.JitteredResize(
            target_size=(IMG_SIZE, IMG_SIZE),
            scale_factor=(0.8, 1.25),
            bounding_box_format="yxyx",
        ),
        keras_cv.layers.MaybeApply(keras_cv.layers.MixUp(), rate=0.5, batchwise=True),
    ]
)

train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(
    eval_resizing,
    num_parallel_calls=tf.data.AUTOTUNE,
)

eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


with strategy.scope():
    model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="yxyx")

    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[12000 * 16 / global_batch, 16000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr, 0.01 * base_lr],
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
    )

weight_decay = 0.0001
step = 0

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.tensorboard_path, write_steps_per_second=True
    ),
    PyCOCOCallback(eval_ds, bounding_box_format="yxyx"),
]
model.compile(
    optimizer=optimizer,
    box_loss="Huber",
    classification_loss="SparseCategoricalCrossentropy",
    rpn_box_loss="Huber",
    rpn_classification_loss="BinaryCrossentropy",
)
model.fit(train_ds, epochs=18, callbacks=callbacks)
