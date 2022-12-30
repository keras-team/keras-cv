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

from random import sample
import sys

import tensorflow as tf
from absl import flags
import matplotlib.pyplot as plt

import keras_cv
from keras_cv.datasets.pascal_voc.segmentation import load
from keras_cv.models.segmentation.deeplab import DeepLabV3
from keras_cv.models.segmentation.resnet_deeplab import DeeplabV3Plus

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

#all_ds = load(split="sbd_train", data_dir=None)
#all_ds = all_ds.concatenate(load(split="sbd_eval", data_dir=None))
#train_ds = all_ds.take(10000)
#eval_ds = all_ds.skip(10000).concatenate(load(split="diff", data_dir=None))
train_ds = load(split="sbd_train", data_dir=None)
eval_ds = load(split="sbd_eval", data_dir=None)

resize_layer = tf.keras.layers.Resizing(512, 512, interpolation="nearest")

image_size = [512, 512, 3]

# TODO: migrate to KPL.
def resize_and_crop_image(
    image,
    desired_size,
    padded_size,
    aug_scale_min=1.0,
    aug_scale_max=1.0,
    seed=1,
    method=tf.image.ResizeMethod.BILINEAR,
):
    with tf.name_scope("resize_and_crop_image"):
        image_size = tf.cast(tf.shape(image)[0:2], tf.float32)

        random_jittering = aug_scale_min != 1.0 or aug_scale_max != 1.0

        if random_jittering:
            random_scale = tf.random.uniform(
                [], aug_scale_min, aug_scale_max, seed=seed
            )
            scaled_size = tf.round(random_scale * desired_size)
        else:
            scaled_size = desired_size

        scale = tf.minimum(
            scaled_size[0] / image_size[0], scaled_size[1] / image_size[1]
        )
        scaled_size = tf.round(image_size * scale)

        # Computes 2D image_scale.
        image_scale = scaled_size / image_size

        # Selects non-zero random offset (x, y) if scaled image is larger than
        # desired_size.
        if random_jittering:
            max_offset = scaled_size - desired_size
            max_offset = tf.where(
                tf.less(max_offset, 0), tf.zeros_like(max_offset), max_offset
            )
            offset = max_offset * tf.random.uniform(
                [
                    2,
                ],
                0,
                1,
                seed=seed,
            )
            offset = tf.cast(offset, tf.int32)
        else:
            offset = tf.zeros((2,), tf.int32)

        scaled_image = tf.image.resize(
            image, tf.cast(scaled_size, tf.int32), method=method
        )

        if random_jittering:
            scaled_image = scaled_image[
                offset[0] : offset[0] + desired_size[0],
                offset[1] : offset[1] + desired_size[1],
                :,
            ]

        output_image = tf.image.pad_to_bounding_box(
            scaled_image, 0, 0, padded_size[0], padded_size[1]
        )

        image_info = tf.stack(
            [
                image_size,
                tf.constant(desired_size, dtype=tf.float32),
                image_scale,
                tf.cast(offset, tf.float32),
            ]
        )
        return output_image, image_info

def resize_and_crop_masks(masks, image_scale, output_size, offset):
  with tf.name_scope('resize_and_crop_masks'):
    mask_size = tf.cast(tf.shape(masks)[1:3], tf.float32)
    num_channels = tf.shape(masks)[3]
    # Pad masks to avoid empty mask annotations.
    masks = tf.concat([
        tf.zeros([1, mask_size[0], mask_size[1], num_channels],
                 dtype=masks.dtype), masks
    ],
                      axis=0)

    scaled_size = tf.cast(image_scale * mask_size, tf.int32)
    scaled_masks = tf.image.resize(
        masks, scaled_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    offset = tf.cast(offset, tf.int32)
    scaled_masks = scaled_masks[
        :,
        offset[0]:offset[0] + output_size[0],
        offset[1]:offset[1] + output_size[1],
        :]

    output_masks = tf.image.pad_to_bounding_box(
        scaled_masks, 0, 0, output_size[0], output_size[1])
    # Remove padding.
    output_masks = output_masks[1::]
    return output_masks

def resize_fn(image, cls_seg):
    image, image_info = resize_and_crop_image(
        image, image_size[:2], image_size[:2], 0.5, 2.0
    )
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    cls_seg += 1
    cls_seg = cls_seg[tf.newaxis, ...]
    cls_seg = resize_and_crop_masks(cls_seg, image_scale, image_size[:2], offset)
    cls_seg = cls_seg[0]
    cls_seg -= 1
    cls_seg = tf.where(
        tf.equal(cls_seg, -1), 255 * tf.ones_like(cls_seg), cls_seg
    )
    return image, cls_seg

def flip_fn(image, cls_seg):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        cls_seg = tf.reverse(cls_seg, axis=[1])
    return image, cls_seg

def proc_train_fn(examples):
    image = examples.pop("image")
    image = tf.cast(image, tf.float32)
    # image = resize_layer(image)
    cls_seg = examples.pop("class_segmentation")
    cls_seg = tf.cast(cls_seg, tf.float32)
    # cls_seg = resize_layer(cls_seg)
    image, cls_seg = resize_fn(image, cls_seg)
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
train_ds = train_ds.batch(global_batch)

eval_ds = eval_ds.map(proc_eval_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(global_batch)

train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(2)

with strategy.scope():
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[30000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr],
    )
    # backbone = keras_cv.models.ResNet50V2(
    #     include_rescaling=True, weights="imagenet", include_top=False,
    #     input_shape=[512, 512, 3]
    # ).as_backbone()
    # model = DeepLabV3(classes=21, backbone=backbone, include_rescaling=True)
    model = DeeplabV3Plus(image_size=512, num_classes=21)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, clipnorm=10.0
    )
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=255)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(ignore_class=255),
        tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ]

# model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.weights_path,
        monitor="val_mean_io_u",
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.tensorboard_path, write_steps_per_second=True
    ),
]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

model.fit(train_ds, epochs=100, validation_data=eval_ds, callbacks=callbacks)
