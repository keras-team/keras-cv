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
"""Utility functions for preprocessing demos."""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


def resize(image, label, img_size=(224, 224), num_classes=10):
    image = tf.image.resize(image, img_size)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}


def load_oxford_dataset(
    name="oxford_flowers102",
    batch_size=64,
    img_size=(224, 224),
    as_supervised=True,
):
    # Load dataset.
    data, ds_info = tfds.load(name, as_supervised=as_supervised, with_info=True)
    train_ds = data["train"]
    num_classes = ds_info.features["label"].num_classes

    # Get tf dataset.
    train_ds = train_ds.map(
        lambda x, y: resize(x, y, img_size=img_size, num_classes=num_classes)
    ).batch(batch_size)
    return train_ds


def visualize_dataset(ds):
    outputs = next(iter(ds.take(1)))
    images = outputs["images"]
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


def gallery_show(images):
    images = images.astype(int)
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.show()


def load_elephant_tensor(output_size=(300, 300)):
    elephants = keras.utils.get_file(
        "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    )
    elephants = keras.utils.load_img(elephants, target_size=output_size)
    elephants = keras.utils.img_to_array(elephants)

    many_elephants = tf.repeat(tf.expand_dims(elephants, axis=0), 9, axis=0)
    return many_elephants
