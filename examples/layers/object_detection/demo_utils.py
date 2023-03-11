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
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv import bounding_box


def preprocess_voc(inputs, format, image_size):
    """mapping function to create batched image and bbox coordinates"""
    inputs["image"] = tf.image.resize(inputs["image"], image_size)
    inputs["objects"]["bbox"] = bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=inputs["image"],
        source="rel_yxyx",
        target=format,
    )
    return {
        "images": inputs["image"],
        "bounding_boxes": inputs["objects"]["bbox"],
    }


def load_voc_dataset(
    bounding_box_format,
    name="voc/2007",
    batch_size=9,
    image_size=(224, 224),
):
    dataset = tfds.load(name, split=tfds.Split.TRAIN, shuffle_files=True)
    dataset = dataset.map(
        lambda x: preprocess_voc(
            x, format=bounding_box_format, image_size=image_size
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.padded_batch(
        batch_size, padding_values={"images": None, "bounding_boxes": -1.0}
    )
    return dataset


def visualize_data(data, bounding_box_format):
    data = next(iter(data))
    images = data["images"]
    bounding_boxes = data["bounding_boxes"]
    output_images = visualize_bounding_boxes(
        images, bounding_boxes, bounding_box_format
    ).numpy()
    gallery_show(output_images)


def visualize_bounding_boxes(image, bounding_boxes, bounding_box_format):
    color = np.array([[255.0, 0.0, 0.0]])
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="rel_yxyx",
        images=image,
    )
    return tf.image.draw_bounding_boxes(image, bounding_boxes, color, name=None)


def gallery_show(images):
    images = images.astype(int)
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.show()
