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

from keras_cv.src.api_export import keras_cv_export


def parse_imagenet_example(img_size, crop_to_aspect_ratio):
    """Function to parse a TFRecord example into an image and label"""

    resizing = None
    if img_size:
        resizing = layers.Resizing(
            width=img_size[0],
            height=img_size[1],
            crop_to_aspect_ratio=crop_to_aspect_ratio,
        )

    def apply(example):
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
        if resizing:
            image = resizing(image)

        # Decode label
        label = (
            tf.cast(tf.reshape(parsed[label_key], shape=()), dtype=tf.int32) - 1
        )
        label = tf.one_hot(label, 1000)

        return image, label

    return apply


@keras_cv_export(
    "keras_cv.datasets.imagenet.load", package="keras_cv.datasets.imagenet"
)
def load(
    split,
    tfrecord_path,
    batch_size=None,
    shuffle=True,
    shuffle_buffer=None,
    reshuffle_each_iteration=False,
    img_size=None,
    crop_to_aspect_ratio=True,
):
    """Loads the ImageNet dataset from TFRecords

    Example:
    ```python
    dataset, ds_info = keras_cv.datasets.imagenet.load(
        split="train", tfrecord_path="gs://my-bucket/imagenet-tfrecords"
    )
    ```

    Args:
        split: the split to load. Should be one of "train" or "validation."
        tfrecord_path: the path to your preprocessed ImageNet TFRecords.
            See keras_cv/datasets/imagenet/README.md for preprocessing
            instructions.
        batch_size: how many instances to include in batches after loading.
            Should only be specified if img_size is specified (so that images
            can be resized to the same size before batching).
        shuffle: whether to shuffle the dataset, defaults to True.
        shuffle_buffer: the size of the buffer to use in shuffling.
        reshuffle_each_iteration: whether to reshuffle the dataset on every
            epoch, defaults to False.
        img_size: the size to resize the images to, defaults to None, indicating
            that images should not be resized.

    Returns:
        tf.data.Dataset containing ImageNet. Each entry is a dictionary
        containing keys {"image": image, "label": label} where images is a
        Tensor of shape [H, W, 3] and label is a Tensor of shape [1000].
    """

    if batch_size is not None and img_size is None:
        raise ValueError(
            "Batching can only be performed if images are resized."
        )

    num_splits = 1024 if split == "train" else 128
    filenames = [
        f"{tfrecord_path}/{split}-{i:05d}-of-{num_splits:05d}"
        for i in range(0, num_splits)
    ]

    dataset = tf.data.TFRecordDataset(
        filenames=filenames, num_parallel_reads=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        parse_imagenet_example(img_size, crop_to_aspect_ratio),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle:
        if not batch_size and not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` "
                "must be provided to `keras_cv.datasets.imagenet.load().`"
            )
        shuffle_buffer = shuffle_buffer or 8 * batch_size
        dataset = dataset.shuffle(
            shuffle_buffer, reshuffle_each_iteration=reshuffle_each_iteration
        )

    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)
