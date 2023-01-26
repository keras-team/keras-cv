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


# TODO(tanzhenyu): provide a TPU compatible NMS decoder.
@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MultiClassNonMaxSuppression(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of an object detection model.

    Arguments:
      bounding_box_format: The format of bounding boxes of input dataset. Refer
        [to the keras.io docs](https://keras.io/api/keras_cv/bounding_box/formats/)
        for more details on supported bounding box formats.
      from_logits: boolean, True means input score is logits, False means confidence.
      iou_threshold: a float value in the range [0, 1] representing the minimum
        IoU threshold for two boxes to be considered same for suppression. Defaults
        to 0.5.
      confidence_threshold: a float value in the range [0, 1]. All boxes with
        confidence below this value will be discarded. Defaults to 0.9.
      max_detections: the maximum detections to consider after nms is applied. A large
        number may trigger significant memory overhead. Defaults to 100.
      max_detections_per_class: the maximum detections to consider per class after
        nms is applied. Defaults to 100.
    """

    def __init__(
        self,
        bounding_box_format,
        from_logits,
        iou_threshold=0.5,
        confidence_threshold=0.9,
        max_detections=100,
        max_detections_per_class=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.from_logits = from_logits
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class
        self.built = True

    def call(self, box_prediction, confidence_prediction):
        """Accepts images and raw predictions, and returns bounding box predictions.

        Args:
            box_prediction: Dense Tensor of shape [batch, boxes, 4] in the
                `bounding_box_format` specified in the constructor.
            confidence_prediction: Dense Tensor of shape [batch, boxes, num_classes].
        """
        target_format = "yxyx"
        if bounding_box.is_relative(self.bounding_box_format):
            target_format = bounding_box.as_relative(target_format)

        box_prediction = bounding_box.convert_format(
            box_prediction,
            source=self.bounding_box_format,
            target=target_format,
        )
        if self.from_logits:
            confidence_prediction = tf.nn.softmax(confidence_prediction)

        box_prediction = tf.expand_dims(box_prediction, axis=-2)
        (
            box_pred,
            confidence_pred,
            class_pred,
            valid_det,
        ) = tf.image.combined_non_max_suppression(
            boxes=box_prediction,
            scores=confidence_prediction,
            max_output_size_per_class=self.max_detections_per_class,
            max_total_size=self.max_detections,
            score_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            clip_boxes=False,
        )
        box_pred = bounding_box.convert_format(
            box_pred,
            source=target_format,
            target=self.bounding_box_format,
        )
        bounding_boxes = {
            "boxes": box_pred,
            "confidence": confidence_pred,
            "classes": class_pred,
            "num_detections": valid_det,
        }
        # this is required to comply with KerasCV bounding box format.
        return bounding_box.mask_invalid_detections(bounding_boxes)

    def get_config(self):
        config = {
            "bounding_box_format": self.bounding_box_format,
            "from_logits": self.from_logits,
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "max_detections_per_class": self.max_detections_per_class,
            "max_detections": self.max_detections,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
