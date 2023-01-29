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
Description: Use KerasCV to train a YoloX_tiny on Pascal VOC 2007.
"""
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import keras

import keras_cv
from keras_cv.callbacks import PyCOCOCallback

EPOCHS = 10

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

# parameters from RetinaNet [paper](https://arxiv.org/abs/1708.02002)


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

augmenter = keras_cv.layers.Augmenter(
    layers=[
        keras_cv.layers.RandomRotation(
            factor=0.03,
            bounding_box_format="xywh",
        ),
        keras_cv.layers.RandomShear(
            x_factor=0.2,
            y_factor=0.2,
            bounding_box_format="xywh",
        ),
        keras_cv.layers.RandomFlip(
            bounding_box_format="xywh",
        ),
        keras_cv.layers.MixUp(alpha=0.5),
        keras_cv.layers.Mosaic(bounding_box_format="xywh"),
    ]
)


def apply_augmenter(images, bounding_boxes):
    inputs = {
        "images": images,
        "bounding_boxes": bounding_boxes,
    }

    outputs = augmenter(inputs)

    return outputs["images"], outputs["bounding_boxes"]


def proc_train_fn(bounding_box_format, img_size):
    def apply(inputs):
        image = inputs["image"]
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
        boxes = inputs["objects"]["bbox"]
        classes = tf.cast(inputs["objects"]["label"], tf.float32)
        bounding_boxes = keras_cv.bounding_box.convert_format(
            boxes, images=image, source="rel_yxyx", target=bounding_box_format
        )
        bounding_boxes = {"boxes": bounding_boxes, "classes": classes}
        return image, bounding_boxes

    return apply


def proc_eval_fn(bounding_box_format, target_size):
    def apply(inputs):
        raw_image = inputs["image"]
        raw_image = tf.cast(raw_image, tf.float32)

        img_size = tf.shape(raw_image)
        height = img_size[0]
        width = img_size[1]

        target_height = tf.cond(
            height > width,
            lambda: float(IMG_SIZE),
            lambda: tf.cast(height / width * IMG_SIZE, tf.float32),
        )
        target_width = tf.cond(
            width > height,
            lambda: float(IMG_SIZE),
            lambda: tf.cast(width / height * IMG_SIZE, tf.float32),
        )
        image = tf.image.resize(
            raw_image, (target_height, target_width), antialias=False
        )

        boxes = keras_cv.bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=image,
            source="rel_yxyx",
            target="xyxy",
        )
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, target_size[0], target_size[1]
        )
        boxes = keras_cv.bounding_box.convert_format(
            boxes,
            images=image,
            source="xyxy",
            target=bounding_box_format,
        )
        classes = tf.cast(inputs["objects"]["label"], tf.float32)
        bounding_boxes = {"boxes": boxes, "classes": classes}
        return image, bounding_boxes

    return apply


def pad_fn(images, bounding_boxes):
    boxes = bounding_boxes["boxes"].to_tensor(
        default_value=-1.0, shape=[GLOBAL_BATCH_SIZE, 32, 4]
    )
    classes = bounding_boxes["classes"].to_tensor(
        default_value=-1.0, shape=[GLOBAL_BATCH_SIZE, 32]
    )
    return images, {"boxes": boxes, "classes": classes}


train_ds = train_ds.map(
    proc_train_fn("xywh", image_size), num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
)
train_ds = train_ds.map(apply_augmenter, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(8 * strategy.num_replicas_in_sync)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

eval_ds = eval_ds.map(
    proc_eval_fn(bounding_box_format="xywh", target_size=image_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
eval_ds = eval_ds.apply(
    tf.data.experimental.dense_to_ragged_batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
)
eval_ds = eval_ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

"""
## Model creation

We'll use the KerasCV API to construct a YoloX_tiny model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

with strategy.scope():
    backbone = keras_cv.models.CSPDarkNetTiny(
        include_rescaling=False, include_top=False, weights="imagenet"
    ).as_backbone(min_level=3)

    model = keras_cv.models.YoloX_tiny(
        # number of classes to be used in box classification
        classes=20,
        # For more info on supported bounding box formats, visit
        # https://keras.io/api/keras_cv/bounding_box/
        bounding_box_format="xywh",
        backbone=backbone,
    )
    # Fine-tuning a YoloX_tiny is as simple as setting backbone.trainable = False
    # model.backbone.trainable = False
    optimizer = tf.optimizers.SGD(learning_rate=BASE_LR, global_clipnorm=10.0)

model.compile(
    classification_loss="binary_crossentropy",
    objectness_loss="binary_crossentropy",
    box_loss="iou",
    optimizer=optimizer,
)

callbacks = [
    # keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_path),
    keras.callbacks.ReduceLROnPlateau(patience=5),
    keras.callbacks.EarlyStopping(patience=10),
    # keras.callbacks.ModelCheckpoint(FLAGS.weights_path, save_weights_only=True),
    # PyCOCOCallback(eval_ds, "xywh"),
]

history = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

"""
# visualization
images, y_true = next(iter(train_ds.take(1)))


from luketils import visualization


class_ids = [
    "Aeroplane",
    "Bicycle",
    "Bird",
    "Boat",
    "Bottle",
    "Bus",
    "Car",
    "Cat",
    "Chair",
    "Cow",
    "Dining Table",
    "Dog",
    "Horse",
    "Motorbike",
    "Person",
    "Potted Plant",
    "Sheep",
    "Sofa",
    "Train",
    "Tvmonitor",
    "Total",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

model.load_weights("ckpt/")
model.prediction_decoder.suppression_layer.confidence_threshold = 0.2
y_pred = model.predict(images)
visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format="xywh",
    y_true=y_true,
    y_pred=None,
    scale=4,
    rows=2,
    cols=2,
    show=True,
    thickness=4,
    font_scale=1,
    class_mapping=class_mapping,
)
"""
