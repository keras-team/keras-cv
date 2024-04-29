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

import warnings

from keras_cv.src import bounding_box
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops


@keras_cv_export("keras_cv.losses.GIoULoss")
class GIoULoss(keras.losses.Loss):
    """Implements the Generalized IoU Loss

    GIoU loss is a modified IoU loss commonly used for object detection. This
    loss aims to directly optimize the IoU score between true boxes and
    predicted boxes. GIoU loss adds a penalty term to the IoU loss that takes in
    account the area of the smallest box enclosing both the boxes being
    considered for the iou. The length of the last dimension should be 4 to
    represent the bounding boxes.

    Args:
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values.For detailed
            information on the supported formats, see the [KerasCV bounding box
            documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        axis: the axis along which to mean the ious, defaults to -1.

    References:
        - [GIoU paper](https://arxiv.org/pdf/1902.09630)
        - [TFAddons Implementation](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/GIoULoss)

    Example:
    ```python
    y_true = np.random.uniform(size=(5, 10, 5), low=0, high=10)
    y_pred = np.random.uniform(size=(5, 10, 4), low=0, high=10)
    loss = GIoULoss(bounding_box_format = "xyWH")
    loss(y_true, y_pred).numpy()
    ```

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.GIoULoss())
    ```
    """  # noqa: E501

    def __init__(self, bounding_box_format, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.axis = axis

    def _compute_enclosure(self, boxes1, boxes2):
        y_min1, x_min1, y_max1, x_max1 = ops.split(boxes1[..., :4], 4, axis=-1)
        y_min2, x_min2, y_max2, x_max2 = ops.split(boxes2[..., :4], 4, axis=-1)
        boxes2_rank = len(boxes2.shape)
        perm = [1, 0] if boxes2_rank == 2 else [0, 2, 1]
        # [N, M] or [batch_size, N, M]
        zeros_t = ops.cast(0, boxes1.dtype)

        enclose_ymin = ops.minimum(y_min1, ops.transpose(y_min2, perm))
        enclose_xmin = ops.minimum(x_min1, ops.transpose(x_min2, perm))
        enclose_ymax = ops.maximum(y_max1, ops.transpose(y_max2, perm))
        enclose_xmax = ops.maximum(x_max1, ops.transpose(x_max2, perm))
        enclose_width = ops.maximum(zeros_t, enclose_xmax - enclose_xmin)
        enclose_height = ops.maximum(zeros_t, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height

        return enclose_area

    def _compute_giou(self, boxes1, boxes2):
        boxes1_rank = len(boxes1.shape)
        boxes2_rank = len(boxes2.shape)

        if boxes1_rank not in [2, 3]:
            raise ValueError(
                "compute_iou() expects boxes1 to be batched, or to be "
                f"unbatched. Received len(boxes1.shape)={boxes1_rank}, "
                f"len(boxes2.shape)={boxes2_rank}. Expected either "
                "len(boxes1.shape)=2 AND or len(boxes1.shape)=3."
            )
        if boxes2_rank not in [2, 3]:
            raise ValueError(
                "compute_iou() expects boxes2 to be batched, or to be "
                f"unbatched. Received len(boxes1.shape)={boxes1_rank}, "
                f"len(boxes2.shape)={boxes2_rank}. Expected either "
                "len(boxes2.shape)=2 AND or len(boxes2.shape)=3."
            )

        target_format = "yxyx"
        if bounding_box.is_relative(self.bounding_box_format):
            target_format = bounding_box.as_relative(target_format)

        boxes1 = bounding_box.convert_format(
            boxes1, source=self.bounding_box_format, target=target_format
        )

        boxes2 = bounding_box.convert_format(
            boxes2, source=self.bounding_box_format, target=target_format
        )

        intersect_area = bounding_box.iou._compute_intersection(boxes1, boxes2)
        boxes1_area = bounding_box.iou._compute_area(boxes1)
        boxes2_area = bounding_box.iou._compute_area(boxes2)
        boxes2_area_rank = len(boxes2_area.shape)
        boxes2_axis = 1 if (boxes2_area_rank == 2) else 0
        boxes1_area = ops.expand_dims(boxes1_area, axis=-1)
        boxes2_area = ops.expand_dims(boxes2_area, axis=boxes2_axis)
        union_area = boxes1_area + boxes2_area - intersect_area
        iou = ops.divide(intersect_area, union_area + keras.backend.epsilon())

        # giou calculation
        enclose_area = self._compute_enclosure(boxes1, boxes2)

        return iou - ops.divide(
            (enclose_area - union_area), enclose_area + keras.backend.epsilon()
        )

    def call(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            raise ValueError(
                "GIoULoss does not support sample_weight. Please ensure "
                f"sample_weight=None. Got sample_weight={sample_weight}"
            )

        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)

        if y_pred.shape[-1] != 4:
            raise ValueError(
                "GIoULoss expects y_pred.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_pred.shape[-1]={y_pred.shape[-1]}."
            )

        if y_true.shape[-1] != 4:
            raise ValueError(
                "GIoULoss expects y_true.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_true.shape[-1]={y_true.shape[-1]}."
            )

        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "GIoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )

        giou = self._compute_giou(y_true, y_pred)
        giou = ops.diagonal(
            giou,
        )
        if self.axis == "no_reduction":
            warnings.warn(
                "`axis='no_reduction'` is a temporary API, and the API "
                "contract will be replaced in the future with a more generic "
                "solution covering all losses."
            )
        else:
            giou = ops.mean(giou, axis=self.axis)

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
