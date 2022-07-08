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


def load_voc_dataset(
    name="voc/2007",
    batch_size=9,
):
    def resize_voc(inputs, img_size=(224, 224)):
        """mapping function to create batched image and bbox coordinates"""
        inputs["image"] = tf.image.resize(inputs["image"], img_size)[0]
        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"][0],
            images=inputs["image"],
            source="rel_yxyx",
            target="rel_xyxy",
        )
        return inputs

    dataset = tfds.load(name, split=tfds.Split.TRAIN, batch_size=1, shuffle_files=True)
    dataset = dataset.map(lambda x: resize_voc(x, img_size=(224, 224)))
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
    return next(iter(dataset.take(1)))


def visualize_data(data, bounding_box_format):
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


def load_elephant_tensor(output_size=(300, 300)):
    elephants = tf.keras.utils.get_file(
        "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    )
    elephants = tf.keras.utils.load_img(elephants, target_size=output_size)
    elephants = tf.keras.utils.img_to_array(elephants)

    many_elephants = tf.repeat(tf.expand_dims(elephants, axis=0), 9, axis=0)
    return many_elephants
