import resource

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

import keras_cv

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

BATCH_SIZE = 8
EPOCHS = 10
CHECKPOINT_PATH = "checkpoint/"

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
class_mapping = dict(zip(range(len(class_ids)), class_ids))

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
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE, drop_remainder=True)
)
eval_ds = eval_ds.map(
    proc_train_fn(bounding_box_format="xywh", img_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE, drop_remainder=True)
)

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""
train_ds = train_ds.prefetch(2)
train_ds = train_ds.shuffle(BATCH_SIZE)
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

We'll use the KerasCV API to construct a YoloX_tiny model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

model = keras_cv.models.YoloX_tiny(
    classes=20,
    bounding_box_format="xywh",
    backbone="cspdarknet",
    include_rescaling=True,
    evaluate_train_time_metrics=False,
)

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
    classification_loss="binary_crossentropy",
    objectness_loss="binary_crossentropy",
    box_loss="iou",
    optimizer=optimizer,
    metrics=metrics,
)

callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
]

history = model.fit(
    train_ds,
    validation_data=eval_ds.take(10),
    epochs=10,
    callbacks=callbacks,
)

"""
## Evaluation
"""

coco_suite = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        max_detections=100,
        name="MaP Standard",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        iou_thresholds=[0.75],
        max_detections=100,
        name="MaP@IoU=0.75",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        iou_thresholds=[0.5],
        max_detections=100,
        name="MaP@IoU=0.5",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        area_range=(0, 32**2),
        max_detections=100,
        name="MaP Small",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        area_range=(32**2, 96**2),
        max_detections=100,
        name="MaP Medium",
    ),
    keras_cv.metrics.COCOMeanAveragePrecision(
        bounding_box_format="xywh",
        class_ids=range(len(class_ids)),
        area_range=(96**2, 1e5**2),
        max_detections=100,
        name="MaP Large",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall Standard",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=1,
        name="Recall MaxDets=1",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=10,
        name="Recall MaxDets=10",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(0, 32**2),
        name="Recall Small",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(32**2, 96**2),
        name="Recall Medium",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(len(class_ids)),
        bounding_box_format="xywh",
        max_detections=100,
        area_range=(96**2, 1e5**2),
        name="Recall Large",
    ),
]
model.compile(
    classification_loss=tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction="none"
    ),
    objectness_loss=tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction="none"
    ),
    box_loss=keras_cv.losses.IoULoss(
        bounding_box_format="center_xywh", mode="squared", reduction="none"
    ),
    optimizer=optimizer,
    metrics=metrics,
)

model.load_weights(CHECKPOINT_PATH)


def proc_eval_fn(bounding_box_format, target_size):
    def apply(inputs):
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
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE, drop_remainder=True)
)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

keras_cv_metrics = model.evaluate(eval_ds, return_dict=True)
print("Metrics:", keras_cv_metrics)
