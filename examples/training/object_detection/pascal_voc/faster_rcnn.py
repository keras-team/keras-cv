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
image_size = [640, 640, 3]
train_ds = tfds.load(
    "voc/2007", split="train+validation", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load(
        "voc/2012",
        split="train+validation",
        with_info=False,
        shuffle_files=True,
    )
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)

with strategy.scope():
    inputs = tf.keras.layers.Input(shape=image_size)
    x = inputs
    x = tf.keras.applications.resnet.preprocess_input(x)

    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_tensor=x, weights="imagenet"
    )

    c2_output, c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in [
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block6_out",
            "conv5_block3_out",
        ]
    ]
    backbone = tf.keras.Model(
        inputs=inputs,
        outputs={2: c2_output, 3: c3_output, 4: c4_output, 5: c5_output},
    )
    # keras_cv backbone gives 2mAP lower result.
    # TODO(ian): should eventually use keras_cv backbone.
    # backbone = keras_cv.models.ResNet50(
    #     include_top=False, weights="imagenet", include_rescaling=False
    # ).as_backbone()
    model = keras_cv.models.FasterRCNN(
        num_classes=20, bounding_box_format="yxyx", backbone=backbone
    )


# TODO (tanzhenyu): migrate to KPL, as this is mostly a duplicate of
# https://github.com/tensorflow/models/blob/master/official/vision/ops/preprocess_ops.py#L138
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


def resize_and_crop_boxes(boxes, image_scale, output_size, offset):
    with tf.name_scope("resize_and_crop_boxes"):
        # Adjusts box coordinates based on image_scale and offset.
        boxes *= tf.tile(tf.expand_dims(image_scale, axis=0), [1, 2])
        boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
        # Clips the boxes.
        boxes = clip_boxes(boxes, output_size)
        return boxes


def clip_boxes(boxes, image_shape):
    if boxes.shape[-1] != 4:
        raise ValueError(
            "boxes.shape[-1] is {:d}, but must be 4.".format(boxes.shape[-1])
        )

    with tf.name_scope("clip_boxes"):
        if isinstance(image_shape, list) or isinstance(image_shape, tuple):
            height, width = image_shape
            max_length = [height, width, height, width]
        else:
            image_shape = tf.cast(image_shape, dtype=boxes.dtype)
            height, width = tf.unstack(image_shape, axis=-1)
            max_length = tf.stack([height, width, height, width], axis=-1)

        clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
        return clipped_boxes


def get_non_empty_box_indices(boxes):
    # Selects indices if box height or width is 0.
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    indices = tf.where(
        tf.logical_and(tf.greater(height, 0), tf.greater(width, 0))
    )
    return indices[:, 0]


def resize_fn(image, gt_boxes, gt_classes):
    image, image_info = resize_and_crop_image(
        image, image_size[:2], image_size[:2], 0.8, 1.25
    )
    gt_boxes = resize_and_crop_boxes(
        gt_boxes, image_info[2, :], image_info[1, :], image_info[3, :]
    )
    indices = get_non_empty_box_indices(gt_boxes)
    gt_boxes = tf.gather(gt_boxes, indices)
    gt_classes = tf.gather(gt_classes, indices)
    return image, gt_boxes, gt_classes


def flip_fn(image, boxes):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        y1, x1, y2, x2 = tf.split(boxes, num_or_size_splits=4, axis=-1)
        boxes = tf.concat([y1, 1.0 - x2, y2, 1.0 - x1], axis=-1)
    return image, boxes


def proc_train_fn(bounding_box_format, img_size):
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        gt_boxes = inputs["objects"]["bbox"]
        image, gt_boxes = flip_fn(image, gt_boxes)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        image, gt_boxes, gt_classes = resize_fn(image, gt_boxes, gt_classes)

        return {
            "images": image,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
        }

    return apply


# TODO(tanzhenyu): consider remove padding while reduce function tracing.
def pad_fn(examples):
    gt_boxes = examples.pop("gt_boxes")
    gt_classes = examples.pop("gt_classes")
    gt_boxes = gt_boxes.to_tensor(
        default_value=-1.0, shape=[global_batch, 32, 4]
    )
    gt_classes = gt_classes.to_tensor(
        default_value=-1.0, shape=[global_batch, 32]
    )
    return examples["images"], {
        "boxes": gt_boxes,
        "classes": gt_classes,
    }


train_ds = train_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(
        global_batch, drop_remainder=True
    )
)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(2)

eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(
        global_batch, drop_remainder=True
    )
)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(2)


with strategy.scope():
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
    tf.keras.callbacks.ModelCheckpoint(
        FLAGS.weights_path, save_weights_only=True
    ),
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
model.fit(train_ds, epochs=18, validation_data=eval_ds, callbacks=callbacks)
