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
Title: Training a KerasCV model for Imagenet Classification
Author: [ianjjohnson](https://github.com/ianjjohnson)
Date created: 2022/07/25
Last modified: 2022/07/25
Description: Use KerasCV to train an image classifier using modern best practices
"""

import sys

import tensorflow as tf
from absl import flags
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import models

"""
## Overview
KerasCV makes training state-of-the-art classification models easy by providing implementations of modern models, preprocessing techniques, and layers.
In this tutorial, we walk through training a model against the Imagenet dataset using Keras and KerasCV.
This tutorial requires you to have KerasCV installed:
```shell
pip install keras-cv
```
"""

"""
## Setup and flags

"""

flags.DEFINE_string(
    "model_name", None, "The name of the model in KerasCV.models to use."
)
flags.DEFINE_string("imagenet_path", None, "Directory from which to load Imagenet.")
flags.DEFINE_string(
    "backup_path", None, "Directory which will be used for training backups."
)
flags.DEFINE_string(
    "weights_path", None, "Directory which will be used to store weight checkpoints."
)
flags.DEFINE_string(
    "tensorboard_path", None, "Directory which will be used to store tensorboard logs."
)
flags.DEFINE_integer("batch_size", 256, "Batch size for training and evaluation.")
flags.DEFINE_boolean(
    "use_xla", True, "Whether or not to use XLA (jit_compile) for training."
)


FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.model_name not in models.__dict__:
    raise ValueError(f"Invalid model name: {FLAGS.model_name}")

CLASSES = 1000
IMAGE_SIZE = (224, 224)
EPOCHS = 250

"""
## Data loading
This guide uses the
[Imagenet dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012).
Note that this requires manual download, and does not work out-of-the-box with TFDS.
To get started, we first load the dataset from a command-line specified directory where ImageNet is stored as TFRecords.
"""


def parse_imagenet_example(example):
    # Read example
    image_key = "image/encoded"
    label_key = "image/class/label"
    keys_to_features = {
        image_key: tf.io.FixedLenFeature((), tf.string, ""),
        label_key: tf.io.FixedLenFeature([], tf.int64, -1),
    }
    parsed = tf.io.parse_single_example(example, keys_to_features)

    # Decode and resize image
    image_bytes = tf.reshape(parsed[image_key], shape=[])
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = layers.Resizing(
        width=IMAGE_SIZE[0], height=IMAGE_SIZE[1], crop_to_aspect_ratio=True
    )(image)

    # Decode label
    label = tf.cast(tf.reshape(parsed[label_key], shape=()), dtype=tf.int32) - 1
    label = tf.one_hot(label, CLASSES)
    return image, label


def load_imagenet_dataset():
    train_filenames = [
        f"{FLAGS.imagenet_path}/train-{i:05d}-of-01024" for i in range(0, 1024)
    ]
    validation_filenames = [
        f"{FLAGS.imagenet_path}/validation-{i:05d}-of-00128" for i in range(0, 128)
    ]

    train_dataset = tf.data.TFRecordDataset(filenames=train_filenames)
    validation_dataset = tf.data.TFRecordDataset(filenames=validation_filenames)

    train_dataset = train_dataset.map(
        parse_imagenet_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_dataset = validation_dataset.map(
        parse_imagenet_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return train_dataset.batch(FLAGS.batch_size), validation_dataset.batch(
        FLAGS.batch_size
    )


train_ds, test_ds = load_imagenet_dataset()


"""
Next, we augment our dataset.
We define a set of augmentation layers and then apply them to our input dataset.
"""

AUGMENT_LAYERS = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.3),
    keras_cv.layers.CutMix(),
]


@tf.function
def augment(img, label):
    inputs = {"images": img, "labels": label}
    for layer in AUGMENT_LAYERS:
        inputs = layer(inputs)
    return inputs["images"], inputs["labels"]


train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
    tf.data.AUTOTUNE
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)


"""
Now we can begin training our model. We begin by loading a model from KerasCV.
Note that we also specify a distribution strategy while creating the model.
Different distribution strategies may be used for different training hardware, as indicated below.
"""

# For TPU training, use tf.distribute.TPUStrategy()
# MirroredStrategy is best for a single machine with multiple GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = eval(f"models.{FLAGS.model_name}")
    model = model(
        include_rescaling=True,
        include_top=True,
        classes=CLASSES,
        input_shape=IMAGE_SIZE + (3,),
    )


"""
Next, we pick an optimizer. Here we use Adam with a constant learning rate.
Note that learning rate will decrease over time due to the ReduceLROnPlateau callback.
"""


optimizer = optimizers.Adam(learning_rate=0.005)


"""
Next, we pick a loss function. We use CategoricalCrossentropy with label smoothing.
"""


loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.1)


"""
Next, we specify the metrics that we want to track. For this example, we track accuracy.
"""

with strategy.scope():
    training_metrics = [metrics.CategoricalAccuracy()]


"""
As a last piece of configuration, we configure callbacks for the method.
We use EarlyStopping, BackupAndRestore, and a model checkpointing callback.
"""


callbacks = [
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=20, min_lr=0.0001
    ),
    callbacks.EarlyStopping(patience=40),
    callbacks.BackupAndRestore(FLAGS.backup_path),
    callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
]


"""
We can now compile the model and fit it to the training dataset.
"""

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=training_metrics,
    jit_compile=FLAGS.use_xla,
)

model.fit(
    train_ds,
    batch_size=FLAGS.batch_size,
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=test_ds,
)
