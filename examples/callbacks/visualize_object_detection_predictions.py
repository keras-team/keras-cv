import tensorflow as tf

import keras_cv


def _create_bounding_box_dataset(num_samples, bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.random.uniform((1, 512, 512, 3), dtype=tf.float32)
    xs = tf.repeat(xs, repeats=num_samples, axis=0)
    y_classes = tf.zeros((num_samples, 1, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [num_samples, 1, 1])
    ys = tf.concat([ys, y_classes], axis=-1)

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, ys


model = keras_cv.models.RetinaNet(
    classes=20,
    # feature_pyramid=FeaturePyramid(),
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
    evaluate_train_time_metrics=True,
)
# Disable all FPN
model.backbone.trainable = False
model.feature_pyramid.trainable = True

optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=[
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=range(20),
            bounding_box_format="xywh",
            name="Mean Average Precision",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(20),
            bounding_box_format="xywh",
            max_detections=100,
            name="Recall",
        ),
    ],
)


x, y = _create_bounding_box_dataset(num_samples=16, bounding_box_format="xywh")
callbacks = [
    keras_cv.callbacks.VisualizeObjectDetectionPredictions(
        x,
        y,
        value_range=(0, 255),
        bounding_box_format="xywh",
        artifacts_dir="artifacts/",
    ),
]

model.fit(
    x,
    y,
    batch_size=8,
    epochs=100,
    callbacks=callbacks,
)
