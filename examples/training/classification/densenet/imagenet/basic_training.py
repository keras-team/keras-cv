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
from absl import flags
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import models

"""
Title: Training a KerasCV model for Imagenet Classification
Author: [ianjjohnson](https://github.com/ianjjohnson)
Date created: 2022/07/25
Last modified: 2022/07/25
Description: Use KerasCV to train an image classifier using modern best practices
"""

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
    "backup_path", None, "Directory which will be used for training backups"
)
flags.DEFINE_string(
    "weights_path", None, "Directory which will be used to store weight checkpoints."
)
flags.DEFINE_string(
    "tensorboard_path", None, "Directory which will be used to store tensorboard logs."
)
flags.DEFINE_string("batch_size", 256, "Batch size for training and evaluation")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

CLASSES = 1000
IMAGE_SIZE = (224, 224)
EPOCHS = 250

"""
## Data loading
This guide uses the
[Imagenet dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012). Note that this requires manual download, and does not work out-of-the-box with TFDS.
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

    return train_dataset.batch(FLAGS.batch_size), validation_dataset.batch(FLAGS.batch_size)


train_ds, test_ds = load_imagenet_dataset()


"""
Next, we augment our dataset. We define a set of augmentation layers and then apply them to our input dataset using the `apply_augmentation` method from our KerasCV training utils.
"""

AUGMENT_LAYERS = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.3),
    keras_cv.layers.RandomCutout(height_factor=0.1, width_factor=0.1),
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
"""


def get_model():
    model = eval(f"models.{FLAGS.model_name}")
    return model(
        include_rescaling=True,
        include_top=True,
        classes=CLASSES,
        input_shape=IMAGE_SIZE + (3,),
    )


"""
Next, we pick an optimizer. Here we use Adam with a constant learning rate.
Note that learning rate will decrease over time due to the ReduceLROnPlateau callback.
"""


def get_optimizer():
    return optimizers.Adam(learning_rate=0.01)


"""
Next, we pick a loss function. Here we use a built-in Keras loss function, so we simply specify it as a string.
"""


def get_loss_fn():
    return losses.CategoricalCrossentropy(label_smoothing=0.1)


"""
Next, we specify the metrics that we want to track. For this example, we track accuracy. Once again, accuracy is a built-in metric in Keras so we can specify it as a string.
"""


def get_metrics():
    return ["accuracy"]


"""
As a last piece of configuration, we configure callbacks for the method. We use EarlyStopping, BackupAndRestore, and a model checkpointing callback.
"""


def get_callbacks():
    return [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=5, min_lr=0.0005
        ),
        callbacks.EarlyStopping(patience=30),
        callbacks.BackupAndRestore(FLAGS.backup_path),
        callbacks.ModelCheckpoint(
            FLAGS.weights_path, save_best_only=True, save_weights_only=True
        ),
        callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
    ]


"""
We can now compile the model and fit it to the training dataset.
"""

with tf.distribute.MirroredStrategy().scope():
    model = get_model()

model.compile(
    optimizer=get_optimizer(),
    loss=get_loss_fn(),
    metrics=get_metrics(),
)

model.fit(
    train_ds,
    batch_size=FLAGS.batch_size,
    epochs=EPOCHS,
    callbacks=get_callbacks(),
    validation_data=test_ds,
)
