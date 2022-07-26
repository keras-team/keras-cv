from loader import load_pascal_voc
import tensorflow as tf
import keras_cv
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras import callbacks as callbacks_lib
import metrics as metrics_lib
from loss import FocalLoss

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
            bounding_box_format='xywh'',
            name="Standard MaP",
        ),
        keras_cv.metrics.COCORecall(
            class_ids=range(20),
            bounding_box_format='xywh'',
            max_detections=100,
            name="Standard Recall",
        ),
    ]
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
