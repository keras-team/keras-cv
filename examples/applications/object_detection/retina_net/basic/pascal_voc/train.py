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
import sys

import tensorflow as tf
import wandb
from absl import flags
from loader import load_pascal_voc
from tensorflow.keras import callbacks as callbacks_lib
from wandb.keras import WandbCallback

import keras_cv

flags.DEFINE_boolean("wandb", False, "Whether or not to use wandb.")
flags.DEFINE_integer("batch_size", 8, "Training and eval batch size.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.wandb:
    wandb.init(project="pascalvoc-retinanet", entity="keras-team-testing")

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="train", batch_size=FLAGS.batch_size
)
val_ds, val_dataset_info = load_pascal_voc(
    bounding_box_format="xywh", split="validation", batch_size=FLAGS.batch_size
)


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

optimizer = tf.optimizers.SGD(
    learning_rate=learning_rate_fn, momentum=0.9, global_clipnorm=10.0
)
# TODO(lukewood): add FocalLoss to KerasCV

# No rescaling
model = keras_cv.applications.RetinaNet(
    num_classes=20,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights="imagenet",
    include_rescaling=True,
)
model.backbone.trainable = False

model.compile(
    optimizer=optimizer,
    loss=keras_cv.applications.RetinaNetLoss(num_classes=20, reduction="sum"),
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
    callbacks_lib.EarlyStopping(patience=30),
]

if FLAGS.wandb:
    callbacks += [
        WandbCallback(save_model=False),
    ]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=callbacks,
)
