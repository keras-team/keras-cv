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


import keras_cv
import tensorflow as tf


def plot_bounding_box_gallery(images, value_range, bounding_box_format, y_true=None, y_pred=None, **kwargs):
    """plots a gallery of images with corresponding bounding box annotations

    Args:
        images: a Tensor or NumPy array containing images to show in the gallery.
        value_range: value range of the images.
        bounding_box_format: the bounding_box_format  the provided bounding boxes are
            in.
        y_true: a Tensor or RaggedTensor representing the ground truth bounding boxes.
        y_pred: a Tensor or RaggedTensor representing the predicted truth bounding
            boxes.
        kwargs: keyword arguments to propagate to
            `keras_cv.visualization.gallery_show()`.
    """
    pred_color = tf.constant(((255.0, 0, 0),))
    true_color = tf.constant(((0, 255.0, 255.0),))
    plotted_images = images

    if y_pred is not None:
        y_pred = keras_cv.bounding_box.convert_format(
            predictions, source=bounding_box_format, target="rel_yxyx", images=images
        )
        if isinstance(y_pred, tf.RaggedTensor):
            y_pred = y_pred.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(plotted_images, y_pred[..., :4], pred_color)

    if y_true is not None:
        y_true = keras_cv.bounding_box.convert_format(
            y_true, source=bounding_box_format, target="rel_yxyx", images=images
        )
        if isinstance(y_true, tf.RaggedTensor):
            y_true = y_true.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(plotted_images, y_true[..., :4], true_color)

    keras_cv.visualization.plot_gallery(plotted_images, value_range, **kwargs)
