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
    if FLAGS.include_probe:
        return image, label
    else:
        return image


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
    ).shuffle(2000, reshuffle_each_iteration=True)
    validation_dataset = validation_dataset.map(
        parse_imagenet_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).shuffle(2000, reshuffle_each_iteration=True)

    return train_dataset.batch(FLAGS.batch_size), validation_dataset.batch(
        FLAGS.batch_size
    )


train_ds, test_ds = load_imagenet_dataset()


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
        include_probe=FLAGS.include_probe,
        classes=CLASSES,
        value_range=(0, 255),
        target_size=IMAGE_SIZE,
    )

    optimizer = optimizers.SGD(learning_rate=FLAGS.initial_learning_rate, momentum=0.9, global_clipnorm=10)
    loss_fn = losses.SimCLRLoss(temperature=0.5, reduction="none")
    probe_loss = keras.losses.CategoricalCrossentropy(reduction="none", from_logits=True)

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
    optimizer=optimizer,
    loss=loss_fn,
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
