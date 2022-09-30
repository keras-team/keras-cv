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

global_batch = 4
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
)

model = keras_cv.models.FasterRCNN(classes=20, bounding_box_format="xyxy")


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
        gt_boxes = keras_cv.bounding_box.convert_format(
            inputs["objects"]["bbox"],
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
    proc_train_fn("xyxy", [256, 256, 3]), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(global_batch, drop_remainder=True)
)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(2)

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

lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[80000], values=[0.001, 0.0001]
)

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
)

weight_decay = 0.001
step = 0


@tf.function
def train_step(examples):
    image, gt_boxes, gt_classes, box_targets, box_weights, cls_targets, cls_weights = (
        examples["images"],
        examples["gt_boxes"],
        examples["gt_classes"],
        examples["rpn_box_targets"],
        examples["rpn_box_weights"],
        examples["rpn_cls_targets"],
        examples["rpn_cls_weights"],
    )
    with tf.GradientTape() as tape:
        outputs = model(image, gt_boxes, gt_classes, training=True)
        rpn_box_pred, rpn_cls_pred = outputs["rpn_box_pred"], outputs["rpn_cls_pred"]
        rpn_reg_loss = tf.reduce_sum(
            rpn_reg_loss_fn(box_targets, rpn_box_pred, box_weights)
        )
        positive_boxes = tf.reduce_sum(box_weights) + 0.01
        # x4 given huber loss reduce_mean on the box dimension
        rpn_reg_loss /= positive_boxes * global_batch * 0.25
        rpn_cls_loss = tf.reduce_sum(
            rpn_cls_loss_fn(cls_targets, rpn_cls_pred, cls_weights)
        )
        rpn_cls_loss /= 256 * global_batch
        rcnn_reg_loss = tf.reduce_sum(
            rcnn_reg_loss_fn(
                outputs["rcnn_box_targets"],
                outputs["rcnn_box_pred"],
                outputs["rcnn_box_weights"],
            )
        )
        positive_rois = tf.reduce_sum(outputs["rcnn_box_weights"]) + 0.01
        rcnn_reg_loss /= positive_rois * global_batch * 0.25
        rcnn_cls_loss = tf.reduce_sum(
            rcnn_cls_loss_fn(
                outputs["rcnn_cls_targets"],
                outputs["rcnn_cls_pred"],
                outputs["rcnn_cls_weights"],
            )
        )
        # 512 is for num_sampled_rois
        rcnn_cls_loss /= 512 * global_batch
        l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(var) for var in model.trainable_variables]
        )
        total_loss = (
            rpn_reg_loss + rpn_cls_loss + rcnn_reg_loss + rcnn_cls_loss + l2_loss
        )
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return rpn_reg_loss, rpn_cls_loss, rcnn_reg_loss, rcnn_cls_loss, total_loss


rpn_reg_loss = 0.0
rpn_cls_loss = 0.0
rcnn_reg_loss = 0.0
rcnn_cls_loss = 0.0
step_size = 500

for epoch in range(50):
    for examples in train_ds:
        (
            step_rpn_reg_loss,
            step_rpn_cls_loss,
            step_rcnn_reg_loss,
            step_rcnn_cls_loss,
            total_loss,
        ) = train_step(examples)
        rpn_reg_loss += step_rpn_reg_loss
        rpn_cls_loss += step_rpn_cls_loss
        rcnn_reg_loss += step_rcnn_reg_loss
        rcnn_cls_loss += step_rcnn_cls_loss
        step += 1
        if step % step_size == 0:
            print(
                "step {} rpn reg loss {}, rpn cls loss {}, rcnn reg loss {}, rcnn cls loss {}".format(
                    step,
                    rpn_reg_loss / step_size,
                    rpn_cls_loss / step_size,
                    rcnn_reg_loss / step_size,
                    rcnn_cls_loss / step_size,
                )
            )
            rpn_reg_loss = 0.0
            rpn_cls_loss = 0.0
            rcnn_reg_loss = 0.0
            rcnn_cls_loss = 0.0
    print("{} finished epoch {}".format(datetime.now(), epoch))
