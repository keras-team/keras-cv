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
"""Utility functions for preprocessing keypoint demos."""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_cv import keypoint


def _preprocess_AFLW2000(
    image,
    keypoints,
    *,
    ragged_keypoints=False,
    img_size=(224, 224),
    keypoint_format="xy"
):
    image = tf.image.resize(image, img_size)
    keypoints = keypoint.convert_format(
        keypoints, source="rel_xy", target=keypoint_format, images=image
    )

    if ragged_keypoints:
        keypoints = tf.RaggedTensor.from_row_lengths(keypoints, [keypoints.shape[0]])

    return {"images": image, "keypoints": keypoints}


def load_AFLW2000_dataset(
    name="aflw2k3d",
    batch_size=64,
    img_size=(224, 224),
    keypoint_format="xy",
    ragged_keypoints=False,
):
    data, ds_info = tfds.load(name, as_supervised=False, with_info=True)
    train_ds = data["train"]

    train_ds = train_ds.map(
        lambda x: _preprocess_AFLW2000(
            x["image"],
            x["landmarks_68_3d_xy_normalized"],
            img_size=img_size,
            ragged_keypoints=ragged_keypoints,
            keypoint_format=keypoint_format,
        )
    ).batch(batch_size)
    return train_ds


def _visualize_keypoints(keypoints, images, keypoint_format="xy"):
    if len(keypoints.shape) == 2:
        # we make a group if no groups are defined
        keypoints = keypoints[None, ...]

    keypoints = keypoint.convert_format(
        keypoints, source=keypoint_format, target="xy", images=images
    )
    for keypoints_group in keypoints:
        plt.scatter(
            keypoints_group[:, 0], keypoints_group[:, 1], marker="+", linewidths=0.5
        )


def visualize_data(ds, keypoint_format="xy"):
    outputs = next(iter(ds.take(1)))
    images = outputs["images"]
    keypoints = outputs.get("keypoints", None)
    plt.figure(figsize=(8, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        if keypoints is not None:
            _visualize_keypoints(
                keypoints[i].numpy(),
                images=images[i],
                keypoint_format=keypoint_format,
            )
    plt.show()
