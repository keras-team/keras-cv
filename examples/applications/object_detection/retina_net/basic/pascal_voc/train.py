import tensorflow as tf
from loader import load_pascal_voc
from loss import FocalLoss
from tensorflow.keras import callbacks as callbacks_lib
from wandb.keras import WandbCallback

import keras_cv
import wandb

wandb.init(project="pascalvoc-retinanet", entity="keras-team-testing")

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="train", batch_size=2
)
val_ds, val_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="validation", batch_size=2
)


def unpackage_dict(inputs):
    return inputs["images"] / 255.0, inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(global_clipnorm=10.0)

# TODO(lukewood): add FocalLoss to KerasCV

# No rescaling
model = keras_cv.applications.RetinaNet(
    num_classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.compile(
    optimizer=optimizer,
    loss=FocalLoss(num_classes=20),
    metrics=[
        keras_cv.metrics.COCOMeanAveragePrecision(
            class_ids=range(20),
            bounding_box_format="xywh",
            name="Standard MaP",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(20),
            bounding_box_format="xywh",
            max_detections=100,
            name="Standard Recall",
        ),
    ],
)

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    WandbCallback(),
    callbacks_lib.EarlyStopping(patience=30),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=callbacks,
)
