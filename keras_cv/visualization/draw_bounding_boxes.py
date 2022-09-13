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

import keras_cv


def draw_bounding_boxes(images, bounding_boxes, color, bounding_box_format):
    """draws bounding boxes on the target image.

    Args:
        images: a batch Tensor of images to plot bounding boxes onto.
        bounding_boxes: a Tensor of batched bounding boxes to plot onto the provided
            images
        color: the color in which to plot the bounding boxes
        bounding_box_format: The format of bounding boxes to plot onto the images. Refer
          [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
          for more details on supported bounding box formats.
    Returns:
        images with bounding boxes plotted on top of them
    """

    color = tf.constant(color)
    color = tf.expand_dims(color, axis=0)

    bounding_boxes = keras_cv.bounding_box.convert_format(
        bounding_boxes, source=bounding_box_format, target="rel_yxyx", images=images
    )

    if isinstance(bounding_boxes, tf.RaggedTensor):
        bounding_boxes = bounding_boxes.to_tensor(default_value=-1)

    return tf.image.draw_bounding_boxes(images, bounding_boxes[..., :4], color)
