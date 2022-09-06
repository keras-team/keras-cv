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
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from keras_cv import losses
from keras_cv import models
from keras_cv import training
from keras_cv.datasets import imagenet

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
flags.DEFINE_float(
    "initial_learning_rate",
    0.1,
    "Initial learning rate which will reduce on plateau.",
)
flags.DEFINE_boolean(
    "include_probe",
    True,
    "Whether to include probing during training.",
)


FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.model_name not in models.__dict__:
    raise ValueError(f"Invalid model name: {FLAGS.model_name}")

CLASSES = 1000
IMAGE_SIZE = (224, 224)
EPOCHS = 250


train_ds = imagenet.load(
    split="train",
    tfrecord_path=FLAGS.imagenet_path,
    batch_size=FLAGS.batch_size,
    img_size=IMAGE_SIZE,
    shuffle=True,
    shuffle_buffer=2000,
    reshuffle_each_iteration=True,
)

# For TPU training, use tf.distribute.TPUStrategy()
# MirroredStrategy is best for a single machine with multiple GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = models.__dict__[FLAGS.model_name]
    model = model(
        include_rescaling=True,
        include_top=False,
        input_shape=IMAGE_SIZE + (3,),
        pooling="avg",
    )
    trainer = training.SimCLRTrainer(
        encoder=model,
        augmenter=training.SimCLRAugmenter(
            value_range=(0, 255), target_size=IMAGE_SIZE
        ),
        probe=layers.Dense(CLASSES, name="linear_probe"),
    )

    optimizer = optimizers.SGD(
        learning_rate=FLAGS.initial_learning_rate, momentum=0.9, global_clipnorm=10
    )
    loss_fn = losses.SimCLRLoss(temperature=0.5, reduction="none")
    probe_loss = keras.losses.CategoricalCrossentropy(
        reduction="none", from_logits=True
    )

with strategy.scope():
    training_metrics = [
        metrics.CategoricalAccuracy(name="probe_accuracy"),
        metrics.TopKCategoricalAccuracy(name="probe_top5_accuracy", k=5),
    ]

training_callbacks = [
    callbacks.EarlyStopping(monitor="probe_accuracy", patience=20),
    callbacks.BackupAndRestore(FLAGS.backup_path),
    callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
]

if FLAGS.include_probe:
    training_callbacks += [
        callbacks.ReduceLROnPlateau(
            monitor="probe_accuracy",
            factor=0.1,
            patience=5,
            min_lr=0.0001,
            min_delta=0.005,
        )
    ]

trainer.compile(
    encoder_optimizer=optimizer,
    encoder_loss=loss_fn,
    probe_optimizer=optimizers.Adam(global_clipnorm=10),
    probe_metrics=training_metrics,
    probe_loss=probe_loss,
    jit_compile=FLAGS.use_xla,
)

trainer.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=training_callbacks,
)
