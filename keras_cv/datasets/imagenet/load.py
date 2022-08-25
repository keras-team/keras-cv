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

import tensorflow as tf
from tensorflow.keras import layers


def parse_imagenet_example(example, img_size, crop_to_aspect_ratio):
    """Function to parse a TFRecord example into an image and label"""
    # Read example
    image_key = "image/encoded"
    label_key = "image/class/label"
    keys_to_features = {
        image_key: tf.io.FixedLenFeature((), tf.string, ""),
        label_key: tf.io.FixedLenFeature([], tf.int64, -1),
    }
    parsed = tf.io.parse_single_example(example, keys_to_features)

    # Decode and resize image
    image_bytes = tf.reshape(parsed[image_key], shape=[])
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = layers.Resizing(
        width=img_size[0], height=img_size[1], crop_to_aspect_ratio=crop_to_aspect_ratio
    )(image)

    # Decode label
    label = tf.cast(tf.reshape(parsed[label_key], shape=()), dtype=tf.int32) - 1
    label = tf.one_hot(label, 1000)

    return image, label


def load(
    split,
    tfrecords_path,
    batch_size=None,
    shuffle=True,
    shuffle_buffer=None,
    reshuffle_each_iteration=False,
    img_size=(512, 512),
    crop_to_aspect_ratio=True,
):
    """Loads the ImageNet dataset from TFRecords

    Usage:
    ```python
    dataset, ds_info = keras_cv.datasets.imagenet.load(
        split="train", tfrecords_path="gs://my-bucket/imagenet-tfrecords"
    )
    ```

    Args:
        split: the split to load.  Should be one of "train" or "validation."
        tfrecords_path: the path to your preprocessed ImageNet TFRecords.
            See keras_cv/datasets/imagenet/README.md for preprocessing instructions.
        batch_size: how many instances to include in batches after loading
        shuffle: whether or not to shuffle the dataset.  Defaults to True.
        shuffle_buffer: the size of the buffer to use in shuffling.
        reshuffle_each_iteration: whether to reshuffle the dataset on every epoch.
            Defaults to False.
        img_size: the size to resize the images to. Defaults to (512, 512).

    Returns:
        tf.data.Dataset containing ImageNet.  Each entry is a dictionary containing
        keys {"image": image, "label": label} where images is a Tensor of shape
        [H, W, 3] and label is a Tensor of shape [1000].
    """

    num_splits = 1024 if split == "train" else 128
    filenames = [
        f"{tfrecords_path}/{split}-{i:05d}-of-{num_splits:05d}"
        for i in range(0, num_splits)
    ]

    dataset = tf.data.TFRecordDataset(filenames=filenames)

    dataset = dataset.map(
        lambda example: parse_imagenet_example(example, img_size, crop_to_aspect_ratio),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        if not batch_size and not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` must be "
                "provided to `keras_cv.datasets.imagenet.load().`"
            )
        shuffle_buffer = shuffle_buffer or 8 * batch_size
        dataset = dataset.shuffle(
            shuffle_buffer, reshuffle_each_iteration=reshuffle_each_iteration
        )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset
