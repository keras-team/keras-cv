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


class GIoULoss(tf.keras.losses.Loss):
    """Implements the GIoU Loss
    GIoU loss is a modified IoU loss commonly used for object detection. This loss aims
    to directly optimize the IoU score between true boxes and predicted boxes. GIoU loss
    adds a penalty term to the IoU loss that takes in account the area of the
    smallest box enclosing both the boxes being considered for the iou. The length of
    the last dimension should be atleast 4 to represent the bounding boxes. While
    this dimension can have more than 4 values, these values will be ignored for the
    calculation of this loss.
    Args:
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by at least these 4 values. The inputs
            may contain additional information such as classes and confidence after
            these 4 values but these values will be ignored while calculating
            this loss. For detailed information on the supported formats, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        axis: the axis along which to mean the ious. Passing the string "no_reduction" implies
            mean across no axes. Defaults to -1.

    References:
        - [GIoU paper](https://arxiv.org/pdf/1902.09630)
        - [TFAddons Implementation](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss)
    Sample Usage:
    ```python
    y_true = tf.random.uniform((5, 10, 5), minval=0, maxval=10, dtype=tf.dtypes.int32)
    y_pred = tf.random.uniform((5, 10, 4), minval=0, maxval=10, dtype=tf.dtypes.int32)
    loss = GIoULoss(bounding_box_format = "xyWH")
    loss(y_true, y_pred).numpy()
    ```
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.GIoULoss())
    ```
    """

    def __init__(self, bounding_box_format, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.axis = axis

    def _compute_giou(self, boxes1, boxes2):
        boxes1_rank = len(boxes1.shape)
        boxes2_rank = len(boxes2.shape)

        if (
            boxes1_rank != boxes2_rank
            or boxes1_rank not in [2, 3]
            or boxes2_rank not in [2, 3]
        ):
            raise ValueError(
                "`GIoULoss` expects both boxes to be batched, or both "
                f"boxes to be unbatched.  Received `len(boxes1.shape)`={boxes1_rank}, "
                f"`len(boxes2.shape)`={boxes2_rank}.  Expected either `len(boxes1.shape)`=2 AND "
                "`len(boxes2.shape)`=2, or `len(boxes1.shape)`=3 AND `len(boxes2.shape)`=3."
            )

        target = bounding_box.preserve_rel(
            target_bounding_box_format="yxyx",
            bounding_box_format=self.bounding_box_format,
        )

        boxes1 = bounding_box.convert_format(
            boxes1, source=self.bounding_box_format, target=target
        )

        boxes2 = bounding_box.convert_format(
            boxes2, source=self.bounding_box_format, target=target
        )

        def compute_giou_for_batch(boxes):
            boxes1, boxes2 = boxes
            zero = tf.convert_to_tensor(0.0, boxes1.dtype)
            boxes1_ymin, boxes1_xmin, boxes1_ymax, boxes1_xmax = tf.unstack(
                boxes1[..., :4], 4, axis=-1
            )
            boxes2_ymin, boxes2_xmin, boxes2_ymax, boxes2_xmax = tf.unstack(
                boxes2[..., :4], 4, axis=-1
            )
            boxes1_width = tf.maximum(zero, boxes1_xmax - boxes1_xmin)
            boxes1_height = tf.maximum(zero, boxes1_ymax - boxes1_ymin)
            boxes2_width = tf.maximum(zero, boxes2_xmax - boxes2_xmin)
            boxes2_height = tf.maximum(zero, boxes2_ymax - boxes2_ymin)
            boxes1_area = boxes1_width * boxes1_height
            boxes2_area = boxes2_width * boxes2_height
            intersect_ymin = tf.maximum(boxes1_ymin, boxes2_ymin)
            intersect_xmin = tf.maximum(boxes1_xmin, boxes2_xmin)
            intersect_ymax = tf.minimum(boxes1_ymax, boxes2_ymax)
            intersect_xmax = tf.minimum(boxes1_xmax, boxes2_xmax)
            intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
            intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
            intersect_area = intersect_width * intersect_height

            union_area = boxes1_area + boxes2_area - intersect_area
            iou = tf.math.divide_no_nan(intersect_area, union_area)

            # giou calculation
            enclose_ymin = tf.minimum(boxes1_ymin, boxes2_ymin)
            enclose_xmin = tf.minimum(boxes1_xmin, boxes2_xmin)
            enclose_ymax = tf.maximum(boxes1_ymax, boxes2_ymax)
            enclose_xmax = tf.maximum(boxes1_xmax, boxes2_xmax)
            enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
            enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
            enclose_area = enclose_width * enclose_height
            return iou - tf.math.divide_no_nan(
                (enclose_area - union_area), enclose_area
            )

        if boxes1_rank == 2:
            return compute_giou_for_batch((boxes1, boxes2))
        else:
            return tf.vectorized_map(compute_giou_for_batch, elems=(boxes1, boxes2))

    def call(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError(
                "GIoULoss does not support sample_weight. Please ensure that sample_weight=None."
                f"got sample_weight={sample_weight}"
            )
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        giou = self._compute_giou(y_true, y_pred)
        if self.axis != "no_reduction":
            giou = tf.reduce_mean(giou, axis=self.axis)

        return 1 - giou

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bounding_box_format": self.bounding_box_format,
                "axis": self.axis,
            }
        )
        return config
