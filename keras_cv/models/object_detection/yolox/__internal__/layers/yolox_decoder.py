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
from tensorflow import keras

import keras_cv.layers as cv_layers
from keras_cv import bounding_box


class DecodePredictions(keras.layers.Layer):
    """Decodes the predictions from YoloX head.

    This layer is similar to the decoding code in `YoloX.compute_losses`. This is
    followed by a bounding box suppression layer.

    Arguments:
        bounding_box_format:  The format of bounding boxes of input dataset. Refer
            [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
            for more details on supported bounding box formats.
        classes: The number of classes to be considered for the classification head.
            One of suppression_layer or classes should be provided. Defaults to None.
        suppression_layer: A `keras.layers.Layer` that follows the same API
            signature of the `keras_cv.layers.NonMaxSuppression` layer.  This layer should
            perform a suppression operation such as NonMaxSuppression, or
            SoftNonMaxSuppression.
    """

    def __init__(
        self, bounding_box_format, classes=None, suppression_layer=None, **kwargs
    ):
        super().__init__(**kwargs)
        if not suppression_layer and not classes:
            raise ValueError(
                "DecodePredictions() requires either `suppression_layer` "
                f"or `classes`.  Received `suppression_layer={suppression_layer} and "
                f"classes={classes}`"
            )
        self.bounding_box_format = bounding_box_format
        self.classes = classes

        self.suppression_layer = suppression_layer or cv_layers.NonMaxSuppression(
            classes=classes,
            bounding_box_format=bounding_box_format,
            confidence_threshold=0.01,
            iou_threshold=0.65,
            max_detections=100,
            max_detections_per_class=100,
        )
        if self.suppression_layer.bounding_box_format != self.bounding_box_format:
            raise ValueError(
                "`suppression_layer` must have the same `bounding_box_format` "
                "as the `DecodePredictions()` layer. "
                "Received `DecodePredictions.bounding_box_format="
                f"{self.bounding_box_format}`, `suppression_layer={suppression_layer}`."
            )
        self.built = True

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)[1:-1]
        # predictions = tf.cast(predictions, dtype=tf.float32)

        batch_size = tf.shape(predictions[0])[0]

        grids = []
        strides = []

        shapes = [x.shape[1:3] for x in predictions]
        predictions = [
            tf.reshape(x, [batch_size, -1, 5 + self.classes]) for x in predictions
        ]
        predictions = tf.concat(predictions, axis=1)

        for i in range(len(shapes)):
            shape_x, shape_y = shapes[i]
            grid_x, grid_y = tf.meshgrid(tf.range(shape_y), tf.range(shape_x))
            grid = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
            shape = grid.shape[:2]

            grids.append(tf.cast(grid, predictions.dtype))
            strides.append(
                tf.ones((shape[0], shape[1], 1))
                * image_shape[0]
                / tf.cast(shape_x, tf.float32)
            )

        grids = tf.concat(grids, axis=1)
        strides = tf.concat(strides, axis=1)

        box_xy = tf.expand_dims(
            (predictions[..., :2] + grids) * strides / image_shape, axis=-2
        )
        box_xy = tf.broadcast_to(
            box_xy, [batch_size, tf.shape(predictions)[1], self.classes, 2]
        )
        box_wh = tf.expand_dims(
            tf.exp(predictions[..., 2:4]) * strides / image_shape, axis=-2
        )
        box_wh = tf.broadcast_to(
            box_wh, [batch_size, tf.shape(predictions)[1], self.classes, 2]
        )

        box_confidence = tf.math.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.math.sigmoid(predictions[..., 5:])

        # create and broadcast classes for every box before nms
        box_classes = tf.expand_dims(tf.range(self.classes, dtype=tf.float32), axis=-1)
        box_classes = tf.broadcast_to(
            box_classes, [batch_size, tf.shape(predictions)[1], self.classes, 1]
        )

        box_scores = tf.expand_dims(box_confidence * box_class_probs, axis=-1)

        outputs = tf.concat([box_xy, box_wh, box_classes, box_scores], axis=-1)
        outputs = tf.reshape(outputs, [batch_size, -1, 6])

        # this conversion is rel_center_xywh to rel_xywh
        # small workaround because rel_center_xywh isn't supported yet
        outputs = bounding_box.convert_format(
            outputs,
            source="center_xywh",
            target="xywh",
            images=images,
        )

        outputs = bounding_box.convert_format(
            outputs,
            source="rel_xywh",
            target=self.suppression_layer.bounding_box_format,
            images=images,
        )
        return self.suppression_layer(outputs, images=images)
