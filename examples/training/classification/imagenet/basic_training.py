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

import numpy as np
import tensorflow as tf
from absl import flags
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

import keras_cv
from keras_cv import models
from keras_cv.datasets import imagenet

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
## Setup, constants and flags
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
flags.DEFINE_integer(
    "batch_size",
    128,
    "Batch size for training and evaluation. This will be multiplied by the number of accelerators in use.",
)
flags.DEFINE_boolean(
    "use_xla", True, "Whether or not to use XLA (jit_compile) for training."
)
flags.DEFINE_boolean(
    "use_mixed_precision",
    False,
    "Whether or not to use FP16 mixed precision for training.",
)
flags.DEFINE_float(
    "initial_learning_rate",
    0.05,
    "Initial learning rate which will reduce on plateau. This will be multiplied by the number of accelerators in use",
)
flags.DEFINE_string(
    "model_kwargs",
    "{}",
    "Keyword argument dictionary to pass to the constructor of the model being trained",
)

flags.DEFINE_string(
    "learning_rate_schedule",
    "ReduceOnPlateau",
    "String denoting the type of learning rate schedule to be used",
)

flags.DEFINE_integer(
    "warmup_steps_percentage",
    10,
    "For how many steps expressed in percentage of total steps should the schedule warm up if we're using the warmup schedule",
)

flags.DEFINE_integer(
    "warmup_hold_steps_percentage",
    10,
    "For how many steps expressed in percentage of total steps should the schedule hold the initial learning rate if we're using the warmup schedule",
)

# An upper bound for number of epochs (this script uses EarlyStopping).
flags.DEFINE_string("epochs", 1000, "Epochs to train for")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

CLASSES = 1000
IMAGE_SIZE = (224, 224)
REDUCE_ON_PLATEAU = "ReduceOnPlateau"
COSINE_DECAY_WITH_WARMUP = "CosineDecayWithWarmup"

if FLAGS.model_name not in models.__dict__:
    raise ValueError(f"Invalid model name: {FLAGS.model_name}")

if FLAGS.use_mixed_precision:
    keras.mixed_precision.set_global_policy("mixed_float16")

"""
We start by detecting the type of accelerators we have available and picking an
appropriate distribution strategy accordingly. We scale our learning rate and
batch size based on the number of accelerators being used.
"""

# Try to detect an available TPU. If none is present, default to MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    if FLAGS.use_mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()
print("Number of accelerators: ", strategy.num_replicas_in_sync)

BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
INITIAL_LEARNING_RATE = FLAGS.initial_learning_rate * strategy.num_replicas_in_sync

"""
## Data loading
This guide uses the
[Imagenet dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012).
Note that this requires manual download and preprocessing. You can find more
information about preparing this dataset at keras_cv/datasets/imagenet/README.md
"""

train_ds = imagenet.load(
    split="train",
    tfrecord_path=FLAGS.imagenet_path,
    batch_size=BATCH_SIZE,
    img_size=IMAGE_SIZE,
)
test_ds = imagenet.load(
    split="validation",
    tfrecord_path=FLAGS.imagenet_path,
    batch_size=BATCH_SIZE,
    img_size=IMAGE_SIZE,
)

"""
Next, we augment our dataset.
We define a set of augmentation layers and then apply them to our input dataset.
"""


AUGMENT_LAYERS = [
    keras_cv.layers.RandomFlip(mode="horizontal"),
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
"""

with strategy.scope():
    model = models.__dict__[FLAGS.model_name]
    model = model(
        include_rescaling=True,
        include_top=True,
        classes=CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        **eval(FLAGS.model_kwargs),
    )

"""
Optional LR schedule with cosine decay instead of ReduceLROnPlateau
TODO: Replace with Core Keras LRWarmup when it's released. This is a temporary solution.
"""


def lr_warmup_cosine_decay(
    global_step, warmup_steps, hold=0, total_steps=0, start_lr=0.0, target_lr=1e-3
):
    # Cosine decay
    # There is no tf.pi :(
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + tf.cos(
                tf.constant(np.pi)
                * (global_step - warmup_steps - hold)
                / float(total_steps - warmup_steps - hold)
            )
        )
    )
    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = tf.where(
            global_step > warmup_steps + hold, learning_rate, target_lr
        )

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate


class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")


# If batched
total_steps = len(train_ds) * FLAGS.epochs
warmup_steps = int(FLAGS.warmup_steps_percentage * total_steps)
schedule = WarmUpCosineDecay(
    start_lr=0.0,
    target_lr=INITIAL_LEARNING_RATE,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    hold=FLAGS.warmup_hold_steps_percentage,
)

"""
Next, we pick an optimizer. Here we use SGD.
Note that learning rate will decrease over time due to the ReduceLROnPlateau callback or with the LRWarmup scheduler.
"""

if FLAGS.learning_rate_schedule == COSINE_DECAY_WITH_WARMUP:
    optimizer = optimizers.SGD(learning_rate=schedule, momentum=0.9, global_clipnorm=10)
else:
    optimizer = optimizers.SGD(
        learning_rate=INITIAL_LEARNING_RATE, momentum=0.9, global_clipnorm=10
    )
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
model_callbacks = [
    callbacks.EarlyStopping(patience=20),
    callbacks.BackupAndRestore(FLAGS.backup_path),
    callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path, write_steps_per_second=True),
]

if FLAGS.learning_rate_schedule == REDUCE_ON_PLATEAU:
    model_callbacks.append(
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=10, min_delta=0.001, min_lr=0.0001
        )
    )


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
    batch_size=BATCH_SIZE,
    epochs=FLAGS.epochs,
    callbacks=model_callbacks,
    validation_data=test_ds,
)
