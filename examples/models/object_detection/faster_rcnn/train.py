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
from datetime import datetime

import tensorflow as tf
import tensorflow_datasets as tfds

import keras_cv

# parameters from FasterRCNN [paper](https://arxiv.org/pdf/1506.01497.pdf)

strategy = tf.distribute.MirroredStrategy()
local_batch = 4
global_batch = local_batch * strategy.num_replicas_in_sync
base_lr = 0.01 * global_batch / 16
image_size = [640, 640, 3]
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)

with strategy.scope():
    model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="yxyx")


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
    indices = tf.where(tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
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
    anchors = model.anchor_generator(image_shape=img_size)
    anchors = tf.concat(tf.nest.flatten(anchors), axis=0)

    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(image)
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
        gt_classes = tf.expand_dims(gt_classes, axis=-1)
        box_targets, box_weights, cls_targets, cls_weights = model.rpn_labeler(
            anchors, gt_boxes, gt_classes
        )
        return {
            "images": image,
            "rpn_box_targets": box_targets,
            "rpn_box_weights": box_weights,
            "rpn_cls_targets": cls_targets,
            "rpn_cls_weights": cls_weights,
            "gt_boxes": gt_boxes,
            "gt_classes": gt_classes,
        }

    return apply


def pad_fn(examples):
    gt_boxes = examples.pop("gt_boxes")
    gt_classes = examples.pop("gt_classes")
    examples["gt_boxes"] = gt_boxes.to_tensor(
        default_value=-1.0, shape=[global_batch, 32, 4]
    )
    examples["gt_classes"] = gt_classes.to_tensor(
        default_value=-1.0, shape=[global_batch, 32, 1]
    )
    return examples


train_ds = train_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(2)

eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(2)


with strategy.scope():
    # The keras huber loss will sum all losses and divide by BS * N * 4, instead we want divide by BS
    # using NONE reduction as also summing creates issues
    rpn_reg_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    rpn_cls_loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    rcnn_reg_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    rcnn_cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

    rpn_reg_metric = tf.keras.metrics.Mean()
    rpn_cls_metric = tf.keras.metrics.Mean()
    rcnn_reg_metric = tf.keras.metrics.Mean()
    rcnn_cls_metric = tf.keras.metrics.Mean()

    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[12000 * 16 / global_batch, 16000 * 16 / global_batch],
        values=[base_lr, 0.1 * base_lr, 0.01 * base_lr],
    )

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
    )

weight_decay = 0.0001
step = 0


def compute_loss(examples, training):
    image, gt_boxes, gt_classes, box_targets, box_weights, cls_targets, cls_weights = (
        examples["images"],
        examples["gt_boxes"],
        examples["gt_classes"],
        examples["rpn_box_targets"],
        examples["rpn_box_weights"],
        examples["rpn_cls_targets"],
        examples["rpn_cls_weights"],
    )
    outputs = model(image, gt_boxes=gt_boxes, gt_classes=gt_classes, training=training)
    rpn_box_pred, rpn_cls_pred = outputs["rpn_box_pred"], outputs["rpn_cls_pred"]
    rpn_reg_loss = tf.reduce_sum(
        rpn_reg_loss_fn(box_targets, rpn_box_pred, box_weights)
    )
    # avoid divide by zero
    # positive_boxes = tf.reduce_sum(box_weights) + 0.01
    # x4 given huber loss reduce_mean on the box dimension
    # rpn_reg_loss /= positive_boxes * 0.25
    rpn_reg_loss /= model.rpn_labeler.samples_per_image * global_batch * 0.25
    rpn_cls_loss = tf.reduce_sum(
        rpn_cls_loss_fn(cls_targets, rpn_cls_pred, cls_weights)
    )
    rpn_cls_loss /= model.rpn_labeler.samples_per_image * global_batch

    rcnn_reg_loss = tf.reduce_sum(
        rcnn_reg_loss_fn(
            outputs["rcnn_box_targets"],
            outputs["rcnn_box_pred"],
            outputs["rcnn_box_weights"],
        )
    )
    # avoid divide by zero
    # positive_rois = tf.reduce_sum(outputs["rcnn_box_weights"]) + 0.01
    # rcnn_reg_loss /= positive_rois * 0.25
    rcnn_reg_loss /= model.roi_sampler.num_sampled_rois * global_batch * 0.25
    rcnn_cls_loss = tf.reduce_sum(
        rcnn_cls_loss_fn(
            outputs["rcnn_cls_targets"],
            outputs["rcnn_cls_pred"],
            outputs["rcnn_cls_weights"],
        )
    )
    # 512 is for num_sampled_rois
    rcnn_cls_loss /= model.roi_sampler.num_sampled_rois * global_batch
    l2_vars = []
    for var in model.trainable_variables:
        if "bn" not in var.name:
            l2_vars.append(var)
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(var) for var in l2_vars])
    total_loss = rpn_reg_loss + rpn_cls_loss + rcnn_reg_loss + rcnn_cls_loss + l2_loss
    return rpn_reg_loss, rpn_cls_loss, rcnn_reg_loss, rcnn_cls_loss, l2_loss, total_loss


@tf.function
def train_step(examples):
    with tf.GradientTape() as tape:
        (
            rpn_reg_loss,
            rpn_cls_loss,
            rcnn_reg_loss,
            rcnn_cls_loss,
            l2_loss,
            total_loss,
        ) = compute_loss(examples, training=True)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return rpn_reg_loss, rpn_cls_loss, rcnn_reg_loss, rcnn_cls_loss, l2_loss, total_loss


@tf.function
def eval_step(examples):
    (
        rpn_reg_loss,
        rpn_cls_loss,
        rcnn_reg_loss,
        rcnn_cls_loss,
        _,
        total_loss,
    ) = compute_loss(examples, training=True)
    # the loss is already divided by global batch, so Mean metrics need to scale back
    num_syncs = strategy.num_replicas_in_sync
    rpn_reg_metric.update_state(rpn_reg_loss * num_syncs)
    rpn_cls_metric.update_state(rpn_cls_loss * num_syncs)
    rcnn_reg_metric.update_state(rcnn_reg_loss * num_syncs)
    rcnn_cls_metric.update_state(rcnn_cls_loss * num_syncs)
    return total_loss


@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_eval_step(dataset_inputs):
    return strategy.run(eval_step, args=(dataset_inputs,))


rpn_reg_loss = 0.0
rpn_cls_loss = 0.0
rcnn_reg_loss = 0.0
rcnn_cls_loss = 0.0
l2_loss = 0.0
step_size = 500

dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
dist_eval_ds = strategy.experimental_distribute_dataset(eval_ds)

for epoch in range(1, 19):
    for examples in dist_train_ds:
        (
            step_rpn_reg_loss,
            step_rpn_cls_loss,
            step_rcnn_reg_loss,
            step_rcnn_cls_loss,
            step_l2_loss,
            total_loss,
        ) = distributed_train_step(examples)
        rpn_reg_loss += step_rpn_reg_loss
        rpn_cls_loss += step_rpn_cls_loss
        rcnn_reg_loss += step_rcnn_reg_loss
        rcnn_cls_loss += step_rcnn_cls_loss
        l2_loss += step_l2_loss
        step += 1
        if step % step_size == 0:
            print(
                "step {} rpn_reg {:.4}, rpn_cls {:.4}, rcnn_reg {:.4}, rcnn_cls {:.4}, l2 {:.4}, pos_rois {:.4}, neg_rois {:.4}".format(
                    step,
                    rpn_reg_loss / step_size,
                    rpn_cls_loss / step_size,
                    rcnn_reg_loss / step_size,
                    rcnn_cls_loss / step_size,
                    l2_loss / step_size,
                    model.roi_sampler._positives.result(),
                    model.roi_sampler._negatives.result(),
                )
            )
            rpn_reg_loss = 0.0
            rpn_cls_loss = 0.0
            rcnn_reg_loss = 0.0
            rcnn_cls_loss = 0.0
            l2_loss = 0.0
            model.roi_sampler._positives.reset_state()
            model.roi_sampler._negatives.reset_state()
    for examples in dist_eval_ds:
        _ = distributed_eval_step(examples)
    print(
        "epoch {} rpn reg loss {:.4}, rpn cls loss {:.4}, rcnn reg loss {:.4}, rcnn cls loss {:.4}".format(
            epoch,
            rpn_reg_metric.result(),
            rpn_cls_metric.result(),
            rcnn_reg_metric.result(),
            rcnn_cls_metric.result(),
        )
    )
    rpn_reg_metric.reset_state()
    rpn_cls_metric.reset_state()
    rcnn_reg_metric.reset_state()
    rcnn_cls_metric.reset_state()
    print(f"{datetime.now()} finished epoch {epoch}", flush=True)

    model.save_weights(f"./weights_{epoch}.h5")
