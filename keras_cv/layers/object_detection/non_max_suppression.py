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

from keras_cv.bounding_box.converters import convert_format


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class NonMaxSuppression(tf.keras.layers.Layer):
    """
    Implements the non-max suppression layer.

    Non-maximal suppression is used to suppress potentially repeated boxes by:
    1) picking the highest ranked boxes
    2) pruning away all boxes that have a high IoU with the chosen boxes.

    References:
        - [Yolo paper](https://arxiv.org/pdf/1506.02640)

    Args:
        num_classes: an integer representing the number of classes that a bounding
            box can belong to.
        bounding_box_format: a case-sensitive string which is one of `"xyxy"`,
            `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`. The
            position and shape of the bounding box will be followed by the class and
            confidence values (in that order). This is required for proper ranking of
            the bounding boxes. Therefore, each bounding box is defined by 6 values.
            For detailed information on the supported format, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        confidence_threshold: a float value in the range [0, 1]. All boxes with
            confidence below this value will be discarded. Defaults to 0.05.
        iou_threshold: a float value in the range [0, 1] representing the minimum
            IoU threshold for two boxes to be considered same for suppression. Defaults
            to 0.5.
        max_detections: the maximum detections to consider after nms is applied. A large
            number may trigger significant memory overhead. Defaults to 100.
        max_detections_per_class: the maximum detections to consider per class after
            nms is applied. Defaults to 100.

    Usage:
    ```python
    images = np.zeros((2, 480, 480, 3), dtype = np.float32)
    ex_boxes = np.array([
                            [
                                [0, 0, 1, 1, 4, 0.9],
                                [0, 0, 2, 3, 4, 0.76],
                                [4, 5, 3, 6, 3, 0.89],
                                [2, 2, 3, 3, 6, 0.04],
                            ],
                            [
                                [0, 0, 5, 6, 4, 0.9],
                                [0, 0, 7, 3, 1, 0.76],
                                [4, 5, 5, 6, 4, 0.04],
                                [2, 1, 3, 3, 7, 0.48],
                            ],
    ], dtype = np.float32)

    nms = NonMaxSuppression(
        num_classes=8,
        bounding_box_format="center_xyWH",
        iou_threshold=0.1
    )

    boxes = nms(boxes, images)
    ```
    """

    def __init__(
        self,
        num_classes,
        bounding_box_format,
        confidence_threshold=0.05,
        iou_threshold=0.5,
        max_detections=100,
        max_detections_per_class=100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bounding_box_format = bounding_box_format
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.max_detections_per_class = max_detections_per_class

    def call(self, predictions, images=None):
        # convert to yxyx for the TF NMS operation
        predictions = convert_format(
            predictions,
            source=self.bounding_box_format,
            target="yxyx",
            images=images,
        )

        # preparing the predictions for TF NMS op
        boxes = tf.expand_dims(predictions[..., :4], axis=2)
        classes = tf.cast(predictions[..., 4], tf.int32)
        scores = predictions[..., 5]

        classes = tf.one_hot(classes, self.num_classes)
        scores = tf.expand_dims(scores, axis=-1) * classes

        # applying the NMS operation
        nmsed_boxes = tf.image.combined_non_max_suppression(
            boxes,
            scores,
            self.max_detections_per_class,
            self.max_detections,
            self.iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )

        # output will be a ragged tensor because num_boxes will change across the batch
        return self._encode_to_ragged(nmsed_boxes, images)

    def _encode_to_ragged(self, nmsed_boxes, images):
        # this TensorArray will hold all the valid detections
        boxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for i in tf.range(tf.shape(nmsed_boxes.nmsed_boxes)[0]):
            num_detections = nmsed_boxes.valid_detections[i]

            # recombining with classes and scores
            boxes_recombined = tf.concat(
                [
                    nmsed_boxes.nmsed_boxes[i][:num_detections],
                    tf.expand_dims(
                        nmsed_boxes.nmsed_classes[i][:num_detections], axis=-1
                    ),
                    tf.expand_dims(
                        nmsed_boxes.nmsed_scores[i][:num_detections], axis=-1
                    ),
                ],
                axis=-1,
            )

            # iterate through the boxes and append it to TensorArray
            for j in range(nmsed_boxes.valid_detections[i]):
                boxes = boxes.write(boxes.size(), boxes_recombined[j])

        # stacking to create a tensor
        boxes = boxes.stack()

        # converting all boxes to the original format
        boxes = convert_format(
            tf.expand_dims(boxes, axis=0),
            source="yxyx",
            target=self.bounding_box_format,
            images=images,
        )[0]

        # using cumulative sum to calculate row_limits for ragged tensor
        row_limits = tf.cumsum(nmsed_boxes.valid_detections)

        # creating the output RaggedTensor by splitting boxes at row_limits
        result = tf.RaggedTensor.from_row_limits(values=boxes, row_limits=row_limits)
        return result

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "bounding_box_format": self.bounding_box_format,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "max_detections_per_class": self.max_detections_per_class,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
