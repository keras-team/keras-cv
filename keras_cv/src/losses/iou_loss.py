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


@keras_cv_export("keras_cv.losses.IoULoss")
class IoULoss(keras.losses.Loss):
    """Implements the IoU Loss

    IoU loss is commonly used for object detection. This loss aims to directly
    optimize the IoU score between true boxes and predicted boxes. The length of
    the last dimension should be 4 to represent the bounding boxes. This loss
    uses IoUs according to box pairs and therefore, the number of boxes in both
    y_true and y_pred are expected to be equal i.e. the i<sup>th</sup>
    y_true box in a batch will be compared the i<sup>th</sup> y_pred box.

    Args:
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values. For detailed
            information on the supported formats, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        mode: must be one of
            - `"linear"`. The loss will be calculated as 1 - iou
            - `"quadratic"`. The loss will be calculated as 1 - iou<sup>2</sup>
            - `"log"`. The loss will be calculated as -ln(iou)
            Defaults to "log".
        axis: the axis along which to mean the ious, defaults to -1.

    References:
        - [UnitBox paper](https://arxiv.org/pdf/1608.01471)

    Example:
    ```python
    y_true = np.random.uniform(size=(5, 10, 5), low=10, high=10)
    y_pred = np.random.uniform(size=(5, 10, 5), low=10, high=10)
    loss = IoULoss(bounding_box_format = "xyWH")
    loss(y_true, y_pred)
    ```

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.IoULoss())
    ```
    """  # noqa: E501

    def __init__(self, bounding_box_format, mode="log", axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.mode = mode
        self.axis = axis

        if self.mode not in ["linear", "quadratic", "log"]:
            raise ValueError(
                "IoULoss expects mode to be one of 'linear', 'quadratic' or "
                f"'log' Received mode={self.mode}, "
            )

    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = ops.cast(y_true, y_pred.dtype)

        if y_pred.shape[-1] != 4:
            raise ValueError(
                "IoULoss expects y_pred.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_pred.shape[-1]={y_pred.shape[-1]}."
            )

        if y_true.shape[-1] != 4:
            raise ValueError(
                "IoULoss expects y_true.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_true.shape[-1]={y_true.shape[-1]}."
            )

        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "IoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )

        iou = bounding_box.compute_iou(y_true, y_pred, self.bounding_box_format)
        # pick out the diagonal for corresponding ious
        iou = ops.diagonal(iou)
        if self.axis == "no_reduction":
            warnings.warn(
                "`axis='no_reduction'` is a temporary API, and the API "
                "contract will be replaced in the future with a more generic "
                "solution covering all losses."
            )
        else:
            iou = ops.mean(iou, axis=self.axis)

        if self.mode == "linear":
            loss = 1 - iou
        elif self.mode == "quadratic":
            loss = 1 - iou**2
        elif self.mode == "log":
            loss = -ops.log(iou)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bounding_box_format": self.bounding_box_format,
                "mode": self.mode,
                "axis": self.axis,
            }
        )
        return config
