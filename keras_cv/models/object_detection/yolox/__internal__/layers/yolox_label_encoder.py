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


class YoloXLabelEncoder(layers.Layer):
    """Transforms the raw labels into targets for training.
    Args:
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats. The YoloX label encoder
            doesn't utilize this argument and it's more of a placeholder to be uniform
            with the rest of the object detection API.
    """

    def __init__(self, bounding_box_format, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format

    def call(self, images, target_boxes):
        """Creates box and classification targets for a batch"""
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "`YoloXLabelEncoder`'s `call()` method does not "
                "support RaggedTensor inputs for the `images` argument.  Received "
                f"`type(images)={type(images)}`."
            )

        if isinstance(target_boxes, tf.RaggedTensor):
            target_boxes = target_boxes.to_tensor(default_value=-1)
        target_boxes = tf.cast(target_boxes, tf.float32)

        return target_boxes
