# Copyright 2023 The KerasCV Authors
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

from keras_cv.backend.keras import backend

image_size = 512
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
mean = tf.constant([0.485, 0.456, 0.406])
std = tf.constant([0.229, 0.224, 0.225])


def normalize(input_image, input_mask):
    input_image = tf.image.convert_image_dtype(input_image, tf.float32)
    input_image = (input_image - mean) / tf.maximum(std, backend.epsilon())
    input_image = input_image / 255
    input_mask -= 1
    return input_image, input_mask


def to_dict(datapoint):
    input_image = tf.image.resize(datapoint["image"], (image_size, image_size))
    input_mask = tf.image.resize(
        datapoint["segmentation_mask"],
        (image_size, image_size),
        method="bilinear",
    )

    input_image, input_mask = normalize(input_image, input_mask)
    input_mask = tf.one_hot(
        tf.squeeze(tf.cast(input_mask, tf.int32), axis=-1), depth=3
    )
    return {"images": input_image, "segmentation_masks": input_mask}


def load_oxford_iiit_pet_dataset():
    data, ds_info = tfds.load("oxford_iiit_pet:3.*.*", with_info=True)
    print("Dataset info: ", ds_info)
    dataset = data["train"]
    return (
        dataset.shuffle(10 * BATCH_SIZE)
        .map(to_dict, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
    )


def display(display_list):
    plt.figure(figsize=(6, 6))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def visualize_dataset(ds):
    for samples in ds.take(1):
        sample_image, sample_mask = (
            samples["images"][0],
            samples["segmentation_masks"][0],
        )
        display([sample_image, sample_mask])
