import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import bounding_box
import os
from luketils import visualization

BATCH_SIZE = 16
EPOCHS = 100
CHECKPOINT_PATH = "checkpoint/"

"""
## Data loading

In this guide, we use the data-loading function: `keras_cv.datasets.pascal_voc.load()`.
KerasCV supports a `bounding_box_format` argument in all components that process
bounding boxes.  To match the KerasCV API style, it is recommended that when writing a
custom data loader, you also support a `bounding_box_format` argument.
This makes it clear to those invoking your data loader what format the bounding boxes
are in.

For example:

```python
train_ds, ds_info = keras_cv.datasets.pascal_voc.load(split='train', bounding_box_format='xywh', batch_size=8)
```

Clearly yields bounding boxes in the format `xywh`.  You can read more about
KerasCV bounding box formats [in the API docs](https://keras.io/api/keras_cv/bounding_box/formats/).

Our data comesloaded into the format
`{"images": images, "bounding_boxes": bounding_boxes}`.  This format is supported in all
KerasCV preprocessing components.

Lets load some data and verify that our data looks as we expect it to.
"""

dataset, dataset_info = keras_cv.datasets.pascal_voc.load(
    split="train", bounding_box_format="xywh", batch_size=9
)

class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]

image_size = [640, 640, 3]
train_ds = tfds.load(
    "voc/2007", split="train+test", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)
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
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        gt_boxes = inputs["objects"]["bbox"]
        image, gt_boxes = flip_fn(image, gt_boxes)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target="yxyx",
        )
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        image, gt_boxes, gt_classes = resize_fn(image, gt_boxes, gt_classes)
        gt_classes = tf.expand_dims(gt_classes, axis=-1)
        bounding_boxes = tf.concat([gt_boxes, gt_classes], axis=-1)
        bounding_boxes = keras_cv.bounding_box.convert_format(
            bounding_boxes, images=image, source="yxyx", target=bounding_box_format
        )
        return {"images": image, "bounding_boxes": bounding_boxes}

    return apply


train_ds = train_ds.map(
    proc_train_fn("xywh", image_size), num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(8, drop_remainder=True)
)
eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="xywh", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(8, drop_remainder=True)
)


def visualize_dataset(dataset, bounding_box_format, path=None):
    example = next(iter(dataset))
    images, boxes = example["images"], example["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=boxes,
        scale=4,
        rows=2,
        cols=4,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
        path=path,
    )


visualize_dataset(train_ds, bounding_box_format="xywh", path="train_ground_truths.png")
visualize_dataset(eval_ds, bounding_box_format="xywh", path="eval_ground_truths.png")
"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""
train_ds = train_ds.prefetch(2)
train_ds = train_ds.shuffle(8)
eval_ds = eval_ds.prefetch(2)


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.
"""

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

model = keras_cv.models.RetinaNet(
    # number of classes to be used in box classification
    classes=20,
    # For more info on supported bounding box formats, visit
    # https://keras.io/api/keras_cv/bounding_box/
    bounding_box_format="xywh",
    # KerasCV offers a set of pre-configured backbones
    backbone="resnet50",
    # Each backbone comes with multiple pre-trained weights
    # These weights match the weights available in the `keras_cv.model` class.
    backbone_weights="imagenet",
    # include_rescaling tells the model whether your input images are in the default
    # pixel range (0, 255) or if you have already rescaled your inputs to the range
    # (0, 1).  In our case, we feed our model images with inputs in the range (0, 255).
    include_rescaling=True,
    # Typically, you'll want to set this to False when training a real model.
    # evaluate_train_time_metrics=True makes `train_step()` incompatible with TPU,
    # and also causes a massive performance hit.  It can, however be useful to produce
    # train time metrics when debugging your model training pipeline.
    evaluate_train_time_metrics=False,
)
# Fine-tuning a RetinaNet is as simple as setting backbone.trainable = False
model.backbone.trainable = False

metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(21),
        bounding_box_format="xywh",
        name="Mean Average Precision",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(21),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]
optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=metrics,
)

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

history = model.fit(
    train_ds,
    validation_data=eval_ds.take(10),
    epochs=100,
    callbacks=callbacks,
)

"""
## Training Process Visualization
"""
metrics = history.history
metrics_to_plot = {
    "Train": metrics["loss"],
    "Validation": metrics["val_loss"],
    "Train Box Loss": metrics["box_loss"],
    "Validation Box Loss": metrics["val_box_loss"],
    "Train Classification Loss": metrics["classification_loss"],
    "Validation Classification Loss": metrics["val_classification_loss"],
}

luketils.visualization.line_plot(
    data=metrics_to_plot,
    title="Loss",
    xlabel="Epochs",
    ylabel="Loss",
    transparent=True,
    path="learning_curve.png",
)

metrics_to_plot = {
    "Mean Average Precision": metrics["val_MaP"],
    "Recall": metrics["val_Recall"],
}
luketils.visualization.line_plot(
    data=metrics_to_plot,
    title="Loss",
    xlabel="Epochs",
    ylabel="Loss",
    transparent=True,
    path="coco_metrics.png",
)

"""
## Inference
"""

images, y_true = next(iter(eval_ds.take(1)))
y_pred = model.predict(images)
visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format="xywh",
    y_true=y_true,
    y_pred=y_pred,
    scale=4,
    rows=2,
    cols=4,
    show=True,
    thickness=4,
    font_scale=1,
    class_mapping=class_mapping,
    path="inference.png",
)

"""
## Evaluation
"""

coco_suite = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        max_detections=100,
        name="MaP Standard",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        iou_thresholds=[0.75],
        max_detections=100,
        name="MaP@IoU=0.75",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        iou_thresholds=[0.5],
        max_detections=100,
        name="MaP@IoU=0.5",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        area_range=(0, 32**2),
        max_detections=100,
        name="MaP Small",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        area_range=(32**2, 96**2),
        max_detections=100,
        name="MaP Medium",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(20),
        area_range=(96**2, 1e5**2),
        max_detections=100,
        name="MaP Large",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall Standard",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=1,
        name="Recall MaxDets=1",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=10,
        name="Recall MaxDets=10",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(0, 32**2),
        name="Recall Small",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(32**2, 96**2),
        name="Recall Medium",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(20),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(96**2, 1e5**2),
        name="Recall Large",
    ),
]
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=coco_suite,
)

model.load_weights(CHECKPOINT_PATH)


def proc_eval_fn(bounding_box_format, target_size):
    def apply(inputs):
        source_id = tf.strings.to_number(
            tf.strings.split(inputs["image/filename"], ".")[0], tf.int64
        )
        raw_image = inputs["image"]
        raw_image = tf.cast(raw_image, tf.float32)

        img_size = tf.shape(raw_image)
        height = img_size[0]
        width = img_size[1]

        target_height = tf.cond(
            height > width,
            lambda: 640.0,
            lambda: tf.cast(height / width * 640.0, tf.float32),
        )
        target_width = tf.cond(
            width > height,
            lambda: 640.0,
            lambda: tf.cast(width / height * 640.0, tf.float32),
        )
        image = tf.image.resize(
            raw_image, (target_height, target_width), antialias=False
        )

        gt_boxes = keras_cv.bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=image,
            source="rel_yxyx",
            target="xyxy",
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, target_size[0], target_size[1]
        )
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="xyxy",
            target=bounding_box_format,
        )

        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        gt_classes = tf.expand_dims(gt_classes, axis=-1)
        bounding_boxes = tf.concat([gt_boxes, gt_classes], axis=-1)
        return image, bounding_boxes

    return apply


eval_ds = tfds.load("voc/2007", split="test", with_info=False, shuffle_files=True)
eval_ds = eval_ds.map(
    proc_eval_fn("xywh", [640, 640, 3]), num_parallel_calls=tf.data.AUTOTUNE
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(8, drop_remainder=True)
)
eval_ds = eval_df.prefetch(2)

keras_cv_metrics = model.evaluate(eval_ds, return_dict=True)
print("Metrics:", keras_cv_metrics)
