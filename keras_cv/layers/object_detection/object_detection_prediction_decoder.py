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

from keras_cv import bounding_box
from keras_cv import layers as cv_layers


class ObjectDetectionPredictionDecoder(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of an object detection model.

    By default, ObjectDetectionPredictionDecoder uses a
    `keras_cv.layers.NonMaxSuppression` layer to perform box pruning.

    Attributes:
      classes: Number of classes in the dataset.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      anchor_generator:
      suppression_layer:
      box_variance:
    """

    def __init__(
        self,
        bounding_box_format,
        anchor_generator,
        classes=None,
        suppression_layer=None,
        box_variance=(0.1, 0.1, 0.2, 0.2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not suppression_layer and not classes:
            raise ValueError(
                "ObjectDetectionPredictionDecoder() requires either `suppression_layer` "
                f"or `classes`.  Received `suppression_layer={suppression_layer} and "
                f"classes={classes}`"
            )
        self.bounding_box_format = bounding_box_format
        self.suppression_layer = self.suppression_layer or cv_layers.NonMaxSuppression(
            classes=classes,
            bounding_box_format=bounding_box_format,
            confidence_threshold=0.05,
            iou_threshold=0.5,
            max_detections=100,
            max_detections_per_class=100,
        )
        if suppression_layer.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "`suppression_layer` must have the same `bounding_box_format` "
                "as the `ObjectDetectionPredictionDecoder()` layer. "
                "Received `ObjectDetectionPredictionDecoder.bounding_box_format="
                f"{self.bounding_box_format}`, `suppression_layer={suppression_layer}`."
            )
        self.anchor_generator = anchor_generator
        self.box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self.box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        return boxes

    def call(self, images, predictions):
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "DecodePredictions() does not support tf.RaggedTensor inputs. "
                f"Received images={images}."
            )

        anchor_boxes = self.anchor_generator(images[0])
        anchor_boxes = tf.concat(list(anchor_boxes.values()), axis=0)
        anchor_boxes = bounding_box.convert_format(
            anchor_boxes,
            source=self.anchor_generator.bounding_box_format,
            target=self.bounding_box_format,
            images=images,
        )
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])

        classes = tf.math.argmax(cls_predictions, axis=-1)
        classes = tf.cast(classes, box_predictions.dtype)
        confidence = tf.math.reduce_max(cls_predictions, axis=-1)

        classes = tf.expand_dims(classes, axis=-1)
        confidence = tf.expand_dims(confidence, axis=-1)

        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
        boxes = tf.concat([boxes, classes, confidence], axis=-1)
        return self.suppression_layer(boxes, images=images)
