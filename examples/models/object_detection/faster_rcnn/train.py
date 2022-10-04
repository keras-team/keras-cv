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

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_visible_devices(physical_devices[:1], "GPU")

# parameters from FasterRCNN [paper](https://arxiv.org/pdf/1506.01497.pdf)

global_batch = 4
image_size = [256, 256, 3]
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
).concatenate(
    tfds.load("voc/2012", split="train+test", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="validation", with_info=False).concatenate(
    tfds.load("voc/2012", split="validation", with_info=False)
)

model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="yxyx")


def flip_fn(image, boxes):
    if tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) > 0.5:
        image = tf.image.flip_left_right(image)
        y1, x1, y2, x2 = tf.split(boxes, num_or_size_splits=4, axis=-1)
        boxes = tf.concat([y1, 1.0 - x2, y2, 1.0 - x1], axis=-1)
    return image, boxes


def proc_train_fn(bounding_box_format, img_size):
    anchors = model.anchor_generator(image_shape=img_size)
    anchors = tf.concat(tf.nest.flatten(anchors), axis=0)
    resizing = tf.keras.layers.Resizing(
        height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
    )

    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet50.preprocess_input(image)
        image = resizing(image)
        gt_boxes = inputs["objects"]["bbox"]
        # image, gt_boxes = flip_fn(image, gt_boxes)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
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
    examples["gt_boxes"] = gt_boxes.to_tensor(default_value=-1.0)
    examples["gt_classes"] = gt_classes.to_tensor(default_value=-1.0)
    return examples


def filter_fn(examples):
    gt_boxes = examples["objects"]["bbox"]
    if tf.shape(gt_boxes)[0] <= 0 or tf.reduce_sum(gt_boxes) < 0:
        return False
    else:
        return True


train_ds = train_ds.filter(filter_fn)
train_ds = train_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(2)

eval_ds = eval_ds.filter(filter_fn)
eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="yxyx", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(2)


# The keras huber loss will sum all losses and divide by BS * N * 4, instead we want divide by BS
# using NONE reduction as also summing creates issues
rpn_reg_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
rpn_cls_loss_fn = tf.keras.losses.BinaryCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)
rcnn_reg_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
rcnn_cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction=tf.keras.losses.Reduction.NONE
)

rpn_reg_metric = tf.keras.metrics.Mean()
rpn_cls_metric = tf.keras.metrics.Mean()
rcnn_reg_metric = tf.keras.metrics.Mean()
rcnn_cls_metric = tf.keras.metrics.Mean()

lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[80000], values=[0.001, 0.0001]
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
)

weight_decay = 0.00001
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
    positive_boxes = tf.reduce_sum(box_weights) + 0.01
    # x4 given huber loss reduce_mean on the box dimension
    rpn_reg_loss /= positive_boxes * 0.25
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
    positive_rois = tf.reduce_sum(outputs["rcnn_box_weights"]) + 0.01
    rcnn_reg_loss /= positive_rois * 0.25
    rcnn_cls_loss = tf.reduce_sum(
        rcnn_cls_loss_fn(
            outputs["rcnn_cls_targets"],
            outputs["rcnn_cls_pred"],
            outputs["rcnn_cls_weights"],
        )
    )
    # 512 is for num_sampled_rois
    rcnn_cls_loss /= model.roi_sampler.num_sampled_rois * global_batch
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(var) for var in model.trainable_variables]
    )
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
    rpn_reg_loss, rpn_cls_loss, rcnn_reg_loss, rcnn_cls_loss, _, _ = compute_loss(
        examples, training=True
    )
    rpn_reg_metric.update_state(rpn_reg_loss)
    rpn_cls_metric.update_state(rpn_cls_loss)
    rcnn_reg_metric.update_state(rcnn_reg_loss)
    rcnn_cls_metric.update_state(rcnn_cls_loss)


rpn_reg_loss = 0.0
rpn_cls_loss = 0.0
rcnn_reg_loss = 0.0
rcnn_cls_loss = 0.0
l2_loss = 0.0
step_size = 500

for epoch in range(40):
    for examples in train_ds:
        (
            step_rpn_reg_loss,
            step_rpn_cls_loss,
            step_rcnn_reg_loss,
            step_rcnn_cls_loss,
            step_l2_loss,
            total_loss,
        ) = train_step(examples)
        rpn_reg_loss += step_rpn_reg_loss
        rpn_cls_loss += step_rpn_cls_loss
        rcnn_reg_loss += step_rcnn_reg_loss
        rcnn_cls_loss += step_rcnn_cls_loss
        l2_loss += step_l2_loss
        step += 1
        if step % step_size == 0:
            print(
                "step {} rpn reg loss {}, rpn cls loss {}, rcnn reg loss {}, rcnn cls loss {}, l2 loss {}".format(
                    step,
                    rpn_reg_loss / step_size,
                    rpn_cls_loss / step_size,
                    rcnn_reg_loss / step_size,
                    rcnn_cls_loss / step_size,
                    l2_loss / step_size,
                )
            )
            rpn_reg_loss = 0.0
            rpn_cls_loss = 0.0
            rcnn_reg_loss = 0.0
            rcnn_cls_loss = 0.0
            l2_loss = 0.0
    for examples in eval_ds:
        eval_step(examples)
    print(
        "epoch {} rpn reg loss {}, rpn cls loss {}, rcnn reg loss {}, rcnn cls loss {}".format(
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
    print("{} finished epoch {}".format(datetime.now(), epoch))

    model.save_weights(f"./weights_{epoch}.h5")
