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

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export


def curry_map_function(bounding_box_format):
    """Mapping function to create batched image and bbox coordinates"""

    def apply(inputs):
        images = inputs["image"]
        bounding_boxes = inputs["objects"]["bbox"]
        labels = inputs["objects"]["label"]
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            images=images,
            source="rel_yxyx",
            target=bounding_box_format,
        )

        bounding_boxes = {"boxes": bounding_boxes, "classes": labels}

        outputs = {"images": images, "bounding_boxes": bounding_boxes}
        return outputs

    return apply


@keras_cv_export("keras_cv.datasets.pascal_voc.load")
def load(
    split,
    bounding_box_format,
    batch_size=None,
    shuffle_files=True,
    shuffle_buffer=None,
    dataset="voc/2007",
):
    """Loads the PascalVOC 2007 dataset.

    Example:
    ```python
    dataset, ds_info = keras_cv.datasets.pascal_voc.load(
        split="train", bounding_box_format="xywh", batch_size=9
    )
    ```

    Args:
        split: the split string passed to the `tensorflow_datasets.load()` call.
            Should be one of "train", "test", or "validation."
        bounding_box_format: the keras_cv bounding box format to load the boxes
            into. For a list of supported formats, please refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        batch_size: how many instances to include in batches after loading
        shuffle_buffer: the size of the buffer to use in shuffling.
        shuffle_files: (Optional) whether to shuffle files, defaults to
            True.
        dataset: (Optional) the PascalVOC dataset to load from. Should be either
            'voc/2007' or 'voc/2012', defaults to 'voc/2007'.

    Returns:
        tf.data.Dataset containing PascalVOC. Each entry is a dictionary
        containing keys {"images": images, "bounding_boxes": bounding_boxes}
        where images is a Tensor of shape [batch, H, W, 3] and bounding_boxes is
        a `tf.RaggedTensor` of shape [batch, None, 5].
    """  # noqa: E501
    if dataset not in ["voc/2007", "voc/2012"]:
        raise ValueError(
            "keras_cv.datasets.pascal_voc.load() expects the `dataset` "
            "argument to be either 'voc/2007' or 'voc/2012', but got "
            f"`dataset={dataset}`."
        )
    dataset, dataset_info = tfds.load(
        dataset, split=split, shuffle_files=shuffle_files, with_info=True
    )
    dataset = dataset.map(
        curry_map_function(bounding_box_format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if shuffle_buffer:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if batch_size is not None:
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)
        )
    return dataset, dataset_info
