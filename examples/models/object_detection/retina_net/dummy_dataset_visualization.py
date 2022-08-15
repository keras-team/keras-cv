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

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.keras import optimizers
import statistics
import keras_cv
from keras_cv import bounding_box
import numpy as np
import matplotlib.pyplot as plt
def main():
    bounding_box_format = 'xywh'
    retina_net = keras_cv.models.RetinaNet(
        classes=1,
        bounding_box_format=bounding_box_format,
        backbone="resnet50",
        backbone_weights=None,
        include_rescaling=False,
    )
    loss = keras_cv.losses.ObjectDetectionLoss(
        classes=1,
        classification_loss=keras_cv.losses.FocalLoss(
            from_logits=True, reduction="none"
        ),
        box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
        reduction="sum",
    )

    retina_net.compile(
        optimizer=optimizers.Adam(),
        loss=loss,
        metrics=[
            keras_cv.metrics.COCOMeanAveragePrecision(
                class_ids=range(1),
                bounding_box_format=bounding_box_format,
                name="MaP",
            ),
            keras_cv.metrics.COCORecall(
                class_ids=range(1),
                bounding_box_format=bounding_box_format,
                name="Recall",
            ),
        ],
    )

    xs, ys = _create_bounding_box_dataset(bounding_box_format)

    while True:
        history = retina_net.fit(x=xs, y=ys, epochs=10)
        metrics = history.history
        metrics = [metrics["loss"], metrics["Recall"], metrics["MaP"]]
        metrics = [statistics.mean(metric) for metric in metrics]
        nonzero = [x != 0.0 for x in metrics]

        predictions = retina_net.inference(xs)
        print('predictions:', predictions)
        visualization = visualize_bounding_boxes(xs, ys, bounding_box_format, color=(255.0, 0, 0))
        visualization = visualize_bounding_boxes(visualization, predictions, bounding_box_format, color=(0, 255.0, 0))
        gallery_show(visualization.numpy ())

        entry = input("continue?  enter c to continue, q to quit.")
        if entry == 'q':
            break


def _create_bounding_box_dataset(bounding_box_format):

    # Just about the easiest dataset you can have, all classes are 0, all boxes are
    # exactly the same.  [1, 1, 2, 2] are the coordinates in xyxy
    xs = tf.ones((10, 512, 512, 3), dtype=tf.float32)
    y_classes = tf.zeros((10, 10, 1), dtype=tf.float32)

    ys = tf.constant([0.25, 0.25, 0.1, 0.1], dtype=tf.float32)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.expand_dims(ys, axis=0)
    ys = tf.tile(ys, [10, 10, 1])
    ys = tf.concat([ys, y_classes], axis=-1)

    ys = keras_cv.bounding_box.convert_format(
        ys, source="rel_xywh", target=bounding_box_format, images=xs, dtype=tf.float32
    )
    return xs, ys


def visualize_bounding_boxes(image, bounding_boxes, bounding_box_format, color=(255.0, 0, 0)):
    color = np.array([color])
    bounding_boxes = bounding_box.convert_format(
        bounding_boxes,
        source=bounding_box_format,
        target="rel_yxyx",
        images=image,
    )
    if isinstance(bounding_boxes, tf.RaggedTensor):
        bounding_boxes = bounding_boxes.to_tensor(default_value=-1)
    return tf.image.draw_bounding_boxes(image, bounding_boxes[..., :4], color, name=None)


def gallery_show(images):
    images = images.astype(int)
    for i in range(9):
        image = images[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.astype("uint8"))
        plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
