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


import tensorflow as tf
from tensorflow.keras import layers

from keras_cv.src import bounding_box


class YoloXLabelEncoder(layers.Layer):
    """Transforms the raw labels into targets for training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, images, box_labels):
        """Creates box and classification targets for a batch"""
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`YoloXLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument. "
                f"Received `type(images)={type(images)}`."
            )

        if box_labels["classes"].get_shape().rank != 2:
            raise ValueError(
                "`YoloXLabelEncoder`'s `call()` method expects a label encoded "
                "`box_labels['classes']` argument of shape "
                "`(batch_size, num_boxes)`. "
                "`Received box_labels['classes'].shape="
                f"{box_labels['classes'].shape}`."
            )

        box_labels = bounding_box.to_dense(box_labels)
        box_labels["classes"] = box_labels["classes"][..., tf.newaxis]

        encoded_box_targets = box_labels["boxes"]
        class_targets = box_labels["classes"]
        return encoded_box_targets, class_targets
