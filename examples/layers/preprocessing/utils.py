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
"""Utility functions to help prepare dataset and visualize preprocessed samples."""
import functools

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


def resize(image, label, img_size=(224, 224), num_classes=10):
    image = tf.image.resize(image, img_size)
    label = tf.one_hot(label, num_classes)
    return image, label


def prepare_dataset(
    name="oxford_flowers102",
    batch_size=64,
    img_size=(224, 224),
    as_supervised=True,
):
    # Load dataset.
    data, ds_info = tfds.load(name, as_supervised=as_supervised, with_info=True)
    train_ds = data["train"]
    num_classes = ds_info.features["label"].num_classes

    # Prepare resize function.
    resize_partial = functools.partial(
        resize, img_size=img_size, num_classes=num_classes
    )

    # Get tf dataset.
    train_ds = train_ds.map(lambda x, y: resize_partial(x, y)).batch(batch_size)
    return train_ds


def apply_augmentation(ds, augmentation_fxn=None):
    ds = ds.map(augmentation_fxn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def visualize_dataset(ds=None, take=1):
    for images, _ in ds.take(take):
        plt.figure(figsize=(8, 8))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()
