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
Last modified: 2023/03/29
Description: Use KerasCV to train a RetinaNet on Pascal VOC 2007.
"""  # noqa: E501
import resource
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from absl import flags
from tensorflow import keras

import keras_cv
from keras_cv.callbacks import PyCOCOCallback

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

flags.DEFINE_integer(
    "epochs",
    100,
    "Number of epochs to run for.",
)

flags.DEFINE_string(
    "weights_name",
    "weights_{epoch:02d}.weights.h5",
    "Directory which will be used to store weight checkpoints.",
)
flags.DEFINE_string(
    "tensorboard_path",
    "logs",
    "Directory which will be used to store tensorboard logs.",
)
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# parameters from RetinaNet [paper](https://arxiv.org/abs/1708.02002)

# Try to detect an available TPU. If none is present, defaults to
# MirroredStrategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    # MirroredStrategy is best for a single machine with one or multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

BATCH_SIZE = 4
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
BASE_LR = 0.005 * GLOBAL_BATCH_SIZE / 16
print("Number of accelerators: ", strategy.num_replicas_in_sync)
print("Global Batch Size: ", GLOBAL_BATCH_SIZE)

IMG_SIZE = 640
image_size = [IMG_SIZE, IMG_SIZE, 3]
train_ds = tfds.load(
    "voc/2007", split="train+validation", with_info=False, shuffle_files=True
)
train_ds = train_ds.concatenate(
    tfds.load(
        "voc/2012",
        split="train+validation",
        with_info=False,
        shuffle_files=True,
    )
)
eval_ds = tfds.load("voc/2007", split="test", with_info=False)


def unpackage_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {
        "images": tf.cast(image, tf.float32),
        "bounding_boxes": bounding_boxes,
    }


train_ds = train_ds.map(
    lambda inputs: unpackage_tfds_inputs(inputs, bounding_box_format="xywh"),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.map(
    lambda inputs: unpackage_tfds_inputs(inputs, bounding_box_format="xywh"),
    num_parallel_calls=tf.data.AUTOTUNE,
)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(
            mode="horizontal", bounding_box_format="xywh"
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640),
            scale_factor=(0.8, 1.25),
            bounding_box_format="xywh",
        ),
    ]
)


rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=2,
    magnitude=0.2,
    rate=0.5,
    magnitude_stddev=0.1,
    geometric=False,
)


def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs


train_ds = train_ds.map(apply_rand_augment)
train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE)
)
train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)


def pad_fn(inputs):
    return inputs["images"], keras_cv.bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


train_ds = train_ds.shuffle(8 * strategy.num_replicas_in_sync)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

eval_resizing = keras_cv.layers.Resizing(
    640, 640, pad_to_aspect_ratio=True, bounding_box_format="xywh"
)
eval_ds = eval_ds.map(
    eval_resizing,
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(tf.data.experimental.dense_to_ragged_batch(BATCH_SIZE))
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model. In this tutorial we
use a pretrained ResNet50 backbone using weights. In order to perform
fine-tuning, we freeze the backbone before training. When
`include_rescaling=True` is set, inputs to the model are expected to be in the
range `[0, 255]`.
"""

with strategy.scope():
    model = keras_cv.models.RetinaNet(
        # number of classes to be used in box classification
        num_classes=20,
        # For more info on supported bounding box formats, visit
        # https://keras.io/api/keras_cv/bounding_box/
        bounding_box_format="xywh",
        backbone=keras_cv.models.ResNet50Backbone.from_preset(
            "resnet50_imagenet"
        ),
    )
    lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[12000 * 16, 16000 * 16],
        values=[BASE_LR, 0.1 * BASE_LR, 0.01 * BASE_LR],
    )
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, global_clipnorm=10.0
    )

model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh", confidence_threshold=0.5, from_logits=True
)

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    metrics=[],
)


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xywh", evaluate_freq=1e9
        )

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in tqdm.tqdm(self.data):
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)
        return logs


callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.EarlyStopping(patience=10),
    keras.callbacks.ModelCheckpoint(FLAGS.weights_name, save_weights_only=True),
    # Temporarily need PyCOCOCallback to verify
    # a 1:1 comparison with the PyMetrics version.
    # Currently, results do not match. I have a feeling this is due
    # to how we are creating the boxes in `BoxCOCOMetrics`
    PyCOCOCallback(eval_ds, bounding_box_format="xywh"),
    keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
]

history = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=FLAGS.epochs,
    callbacks=callbacks,
)
