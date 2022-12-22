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
Author: [lukewood](https://github.com/LukeWood), [tanzhenyu](https://github.com/tanzhenyu)
Date created: 2022/09/27
Last modified: 2022/12/08
Description: Use KerasCV to train a RetinaNet on Pascal VOC 2007.
"""
import resource
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import keras

import keras_cv
from keras_cv import bounding_box
from keras_cv import layers

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

EPOCHS = 100
CHECKPOINT_PATH = "checkpoint/"

flags.DEFINE_string(
    "weights_name",
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

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
BASE_LR = 0.01 * GLOBAL_BATCH_SIZE / 16
print("Number of accelerators: ", strategy.num_replicas_in_sync)
print("Global Batch Size: ", GLOBAL_BATCH_SIZE)

IMG_SIZE = 640
image_size = [IMG_SIZE, IMG_SIZE, 3]
train_ds = tfds.load(
    "voc/2007", split="train+validation", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load("voc/2012", split="train+validation", with_info=False, shuffle_files=True)
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)


def unpackage_inputs(bounding_box_format):
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        gt_boxes = tf.cast(inputs["objects"]["bbox"], tf.float32)
        gt_classes = tf.cast(inputs["objects"]["label"], tf.float32)
        gt_boxes = keras_cv.bounding_box.convert_format(
            gt_boxes,
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
        )
        return {
            "images": image,
            "bounding_boxes": {"boxes": gt_boxes, "classes": gt_classes},
        }

    return apply


train_ds = train_ds.map(unpackage_inputs("xywh"), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
)

train_ds = train_ds.shuffle(8 * strategy.num_replicas_in_sync)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

eval_ds = eval_ds.map(
    unpackage_inputs("xywh"),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)


"""
Our data pipeline is now complete.  We can now move on to data augmentation:
"""
eval_resizing = layers.Resizing(
    IMG_SIZE, IMG_SIZE, bounding_box_format="xywh", pad_to_aspect_ratio=True
)

augmenter = layers.Augmenter(
    [
        layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
        layers.JitteredResize(
            target_size=(IMG_SIZE, IMG_SIZE),
            scale_factor=(0.8, 1.25),
            bounding_box_format="xywh",
        ),
        layers.MaybeApply(layers.MixUp(), rate=0.5, batchwise=True),
    ]
)

train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(
    eval_resizing,
    num_parallel_calls=tf.data.AUTOTUNE,
)
"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""


def unpackage_inputs(data):
    return data["images"], data["bounding_boxes"]


train_ds = train_ds.map(unpackage_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_inputs, num_parallel_calls=tf.data.AUTOTUNE)


# TODO(lukewood): the boxes loses shape from KPL, so need to pad to a known shape.
# TODO(tanzhenyu): consider remove padding while reduce function tracing.
def pad_fn(image, boxes):
    return image, bounding_box.to_dense(boxes)


train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)


with strategy.scope():
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
    )
# Fine-tuning a RetinaNet is as simple as setting backbone.trainable = False
model.backbone.trainable = False
optimizer = tf.optimizers.SGD(learning_rate=BASE_LR, global_clipnorm=10.0)

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    metrics=[],
)


callbacks = [
    keras.callbacks.TensorBoard(log_dir="logs"),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=5),
    keras.callbacks.EarlyStopping(monitor="loss", patience=10),
    keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_weights_only=True),
    keras_cv.callbacks.PyCOCOCallback(eval_ds.take(2), "xywh"),
]

history = model.fit(
    train_ds.take(2),
    epochs=50,
    callbacks=callbacks,
)
