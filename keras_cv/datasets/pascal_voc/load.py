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
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv import bounding_box


def curry_map_function(bounding_box_format, img_size):
    """Mapping function to create batched image and bbox coordinates"""

    if img_size is not None:
        resizing = keras.layers.Resizing(
            height=img_size[0], width=img_size[1], crop_to_aspect_ratio=False
        )

    # TODO(lukewood): update `keras.layers.Resizing` to support bounding boxes.
    def apply(inputs):
        # Support image size none.
        if img_size is not None:
            inputs["image"] = resizing(inputs["image"])

        inputs["objects"]["bbox"] = bounding_box.convert_format(
            inputs["objects"]["bbox"],
            images=inputs["image"],
            source="rel_yxyx",
            target=bounding_box_format,
        )

        bounding_boxes = inputs["objects"]["bbox"]
        labels = tf.cast(inputs["objects"]["label"], tf.float32)
        labels = tf.expand_dims(labels, axis=-1)
        bounding_boxes = tf.concat([bounding_boxes, labels], axis=-1)
        return {"images": inputs["image"], "bounding_boxes": bounding_boxes}

    return apply


def load(
    split,
    bounding_box_format,
    batch_size=None,
    shuffle=None,
    shuffle_buffer=None,
    img_size=None,
):
    """Loads the PascalVOC 2007 dataset.

    Usage:
    ```python
    dataset, ds_info = keras_cv.datasets.pascal_voc.load(
        split="train", bounding_box_format="xywh", batch_size=9
    )
    ```

    Args:
        split: the split string passed to the `tensorflow_datasets.load()` call.  Should
            be one of "train", "test", or "validation."
        bounding_box_format: the keras_cv bounding box format to load the boxes into.
            For a list of supported formats, please  Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        batch_size: (Optional) how many instances to include in batches after loading. If
            not provided, no batching will occur.
        shuffle: whether or not to shuffle the dataset.  Defaults to None, which indicates
            shuffling will only occur if a `shuffle_buffer` is provided.
        shuffle_buffer: the size of the buffer to use in shuffling.
        img_size: (Optional) size to resize the images to.  By default, images are not
            resized and ragged batches are produced if batching occurs.

    Returns:
        tf.data.Dataset containing PascalVOC.  Each entry is a dictionary containing
        keys {"images": images, "bounding_boxes": bounding_boxes} where images is a
        Tensor of shape [batch, H, W, 3] and bounding_boxes is a `tf.RaggedTensor` of
        shape [batch, None, 5].
    """
    dataset, dataset_info = tfds.load(
        "voc/2007", split=split, shuffle_files=shuffle, with_info=True
    )
    dataset = dataset.map(
        curry_map_function(bounding_box_format=bounding_box_format, img_size=img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle is None and shuffle_buffer is not None:
        shuffle = True

    if shuffle:
        if not shuffle_buffer:
            raise ValueError(
                "If `shuffle=True`, either a `batch_size` or `shuffle_buffer` must be "
                "provided to `keras_cv.datasets.pascal_voc.load().`"
            )
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if batch_size is not None:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
        )
    return dataset, dataset_info
