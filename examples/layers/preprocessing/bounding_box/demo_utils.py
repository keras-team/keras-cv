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


def load_voc_dataset(
    name="voc/2007",
    batch_size=9,
    image_size=(224, 224),
):
    def resize_voc(inputs):
        """mapping function to create batched image and bbox coordinates"""
        inputs["image"] = tf.image.resize(inputs["image"], image_size)[0]
        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"][0],
            images=inputs["image"],
            source="rel_yxyx",
            target="rel_xyxy",
        )
        return inputs

    dataset = tfds.load(name, split=tfds.Split.TRAIN, batch_size=1, shuffle_files=True)
    dataset = dataset.map(lambda x: resize_voc(x))
    dataset = dataset.padded_batch(
        batch_size,
        padding_values={
            "image": None,
            "labels": None,
            "image/filename": None,
            "labels_no_difficult": None,
            "objects": {
                "bbox": tf.cast(-1, tf.float32),
                "is_difficult": None,
                "is_truncated": None,
                "label": None,
                "pose": None,
            },
        },
    )
    dataset = dataset.map(lambda x: package_to_dict(x))
    return dataset


def visualize_data(data, bounding_box_format):
    data = next(iter(data.take(9)))
    images = data["images"]
    bounding_boxes = data["bounding_boxes"]
    output_images = visualize_bounding_box(
        images, bounding_boxes, bounding_box_format
    ).numpy()
    gallery_show(output_images)


def visualize_bounding_box(image, bounding_boxes, bounding_box_format):
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


def package_to_dict(dataset):
    outputs = {"images": dataset["image"], "bounding_boxes": dataset["objects"]["bbox"]}
    return outputs
