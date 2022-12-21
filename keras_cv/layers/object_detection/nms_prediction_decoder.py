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


class NmsPredictionDecoder(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of an object detection model.

    By default, NmsPredictionDecoder uses a
    `keras_cv.layers.NonMaxSuppression` layer to perform box pruning.  The layer may
    optionally take a `suppression_layer`, which can perform an alternative suppression
    operation, such as SoftNonMaxSuppression.

    Arguments:
      classes: Number of classes in the dataset.
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      anchor_generator: a `keras_cv.layers.AnchorGenerator`.
      suppression_layer: (Optional) a `keras.layers.Layer` that follows the same API
        signature of the `keras_cv.layers.NonMaxSuppression` layer.  This layer should
        perform a suppression operation such as NonMaxSuppression, or
        SoftNonMaxSuppression.
      box_variance: (Optional) The scaling factors used to scale the bounding box
        targets.  Defaults to `(0.1, 0.1, 0.2, 0.2)`.  **Important Note:**
        `box_variance` is applied to the boxes in `xywh` format.
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
                "NmsPredictionDecoder() requires either `suppression_layer` "
                f"or `classes`.  Received `suppression_layer={suppression_layer} and "
                f"classes={classes}`"
            )
        self.bounding_box_format = bounding_box_format
        self.suppression_layer = suppression_layer or cv_layers.NonMaxSuppression(
            classes=classes,
            bounding_box_format=bounding_box_format,
            confidence_threshold=0.5,
            iou_threshold=0.5,
            max_detections=100,
            max_detections_per_class=100,
        )
        if self.suppression_layer.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "`suppression_layer` must have the same `bounding_box_format` "
                "as the `NmsPredictionDecoder()` layer. "
                "Received `NmsPredictionDecoder.bounding_box_format="
                f"{self.bounding_box_format}`, `suppression_layer={suppression_layer}`."
            )
        self.anchor_generator = anchor_generator
        self.box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)
        self.built = True

    def call(self, images, predictions):
        """Accepts images and raw predictions, and returns bounding box predictions.

        Args:
            images: Tensor of shape [batch, height, width, channels].
            predictions: Dense Tensor of shape [batch, anchor_boxes, 6] in the
                `bounding_box_format` specified in the constructor.
        """
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
            target="xywh",
            images=images[0],
        )
        predictions = bounding_box.convert_format(
            predictions, source=self.bounding_box_format, target="xywh", images=images
        )
        box_predictions = predictions['boxes']
        cls_predictions = tf.nn.sigmoid(predictions['classes'])

        classes = tf.math.argmax(cls_predictions, axis=-1)
        classes = tf.cast(classes, box_predictions.dtype)
        confidence = tf.math.reduce_max(cls_predictions, axis=-1)

        boxes = bounding_box._decode_deltas_to_boxes(
            anchors=anchor_boxes[None, ...],
            boxes_delta=box_predictions,
            anchor_format="xywh",
            box_format="xywh",
            variance=self.box_variance,
        )
        boxes = {"boxes": boxes, "classes": classes, "confidence": confidence}
        boxes = bounding_box.convert_format(
            boxes,
            source="xywh",
            target=self.suppression_layer.bounding_box_format,
            images=images,
        )
        # Note: suppression_layer must have same bounding_box_format
        return self.suppression_layer(boxes, images=images)
