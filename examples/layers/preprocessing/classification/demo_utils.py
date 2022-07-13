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

from keras_cv import keypoint


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


def _resize_keypoints(
    image,
    keypoints,
    *,
    ragged_keypoints=False,
    img_size=(224, 224),
    keypoint_format="rel_xy"
):
    image = tf.image.resize(image, img_size)
    keypoints = keypoint.convert_format(
        keypoints, source=keypoint_format, target="xy", images=image
    )
    if ragged_keypoints:
        keypoints = tf.RaggedTensor.from_row_lengths(keypoints, [keypoints.shape[0]])

    return {"images": image, "keypoints": keypoints}


def load_AFLW2000_dataset(
    name="aflw2k3d",
    batch_size=64,
    img_size=(224, 224),
    ragged_keypoints=False,
):
    data, ds_info = tfds.load(name, as_supervised=False, with_info=True)
    train_ds = data["train"]

    train_ds = train_ds.map(
        lambda x: _resize_keypoints(
            x["image"],
            x["landmarks_68_3d_xy_normalized"],
            img_size=img_size,
            ragged_keypoints=ragged_keypoints,
        )
    ).batch(batch_size)
    return train_ds


def _visualize_boxes(boxes):
    # TODO: implement
    pass


def _visualize_keypoints(keypoints):
    if len(keypoints.shape) == 2:
        # we make a group if no groups are defined
        keypoints = keypoints[None, ...]
    for keypoints_group in keypoints:
        plt.scatter(
            keypoints_group[:, 0], keypoints_group[:, 1], marker="+", linewidths=0.5
        )


def visualize_dataset(ds):
    outputs = next(iter(ds.take(1)))
    images = outputs["images"]
    keypoints = outputs.get("keypoints", None)
    bounding_boxes = outputs.get("bounding_boxes", None)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        if bounding_boxes is not None:
            _visualize_boxes(bounding_boxes[i].numpy())
        if keypoints is not None:
            _visualize_keypoints(keypoints[i].numpy())
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
    elephants = tf.keras.utils.get_file(
        "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
    )
    elephants = tf.keras.utils.load_img(elephants, target_size=output_size)
    elephants = tf.keras.utils.img_to_array(elephants)

    many_elephants = tf.repeat(tf.expand_dims(elephants, axis=0), 9, axis=0)
    return many_elephants
