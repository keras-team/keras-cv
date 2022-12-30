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

# Requires TF 2.11+
import tensorflow as tf

import keras_cv.layers.preprocessing as preprocessing
from keras_cv.datasets.pascal_voc.segmentation import load
from keras_cv.models.segmentation.fcn import FCN8S

train_ds = load(split="sbd_train", data_dir=None)
eval_ds = load(split="sbd_eval", data_dir=None)


def resize_image(img, cls_seg, augment=False):
    img = tf.keras.layers.Resizing(224, 224, interpolation="nearest")(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    cls_seg = tf.keras.layers.Resizing(224, 224, interpolation="nearest")(cls_seg)
    cls_seg = tf.cast(cls_seg, tf.int32)

    inputs = {"images": img, "segmentation_masks": cls_seg}
    if augment:
        inputs = preprocessing.RandomFlip("horizontal")(inputs)
        inputs = preprocessing.RandomColorDegeneration(factor=0.3)(inputs)
        inputs = preprocessing.RandomGaussianBlur(kernel_size=24, factor=0.3)(inputs)
        inputs = preprocessing.GridMask(ratio_factor=(0, 0.5))(inputs)
        inputs = preprocessing.RandomRotation(factor=0.1, segmentation_classes=21)(
            inputs
        )

    return inputs["images"], inputs["segmentation_masks"]


def process(examples, augment=False):
    image = examples.pop("image")
    cls_seg = examples.pop("class_segmentation")

    image, cls_seg = resize_image(image, cls_seg, augment=augment)

    sample_weight = tf.equal(cls_seg, 255)
    zeros = tf.zeros_like(cls_seg)
    cls_seg = tf.where(sample_weight, zeros, cls_seg)
    return image, cls_seg


train_ds = (
    train_ds.map(
        lambda x: process(x, augment=False), num_parallel_calls=tf.data.AUTOTUNE
    )
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
train_ds = train_ds.batch(16, drop_remainder=True)

eval_ds = eval_ds.map(lambda x: process(x), num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.batch(16, drop_remainder=True)

train_ds = train_ds.shuffle(8)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

global_batch = 16

base_lr = 0.007 * global_batch / 16

lr_decay = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=[30000 * 16 / global_batch], values=[base_lr, 0.1 * base_lr]
)

optimizer_options = {
    "adam": tf.keras.optimizers.Adam(learning_rate=1e-4),
    "weight_decay": tf.keras.optimizers.Adam(
        learning_rate=1e-4, weight_decay=2e-4, jit_compile=True, amsgrad=True
    ),
    "paper_optimizer": tf.keras.optimizers.SGD(
        learning_rate=lr_decay, momentum=0.9, clipnorm=10.0
    ),
}

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="/fcn_model/",
        monitor="val_mean_io_u",
        save_best_only=True,
        save_weights_only=True,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir="/tensorboard_logs/", write_steps_per_second=True
    ),
]

backbone = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
model = FCN8S(classes=21, backbone=backbone)

model.summary(expand_nested=True)

tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=True,
    show_layer_activations=True,
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=21)
metrics = [
    tf.keras.metrics.SparseCategoricalCrossentropy(ignore_class=21),
    tf.keras.metrics.MeanIoU(num_classes=21, sparse_y_pred=False),
    tf.keras.metrics.SparseCategoricalAccuracy(),
]

model.compile(optimizer=optimizer_options["adam"], loss=loss_fn, metrics=metrics)

history = model.fit(train_ds, epochs=10, validation_data=eval_ds, callbacks=callbacks)
