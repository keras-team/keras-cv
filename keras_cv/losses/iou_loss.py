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
from keras_cv.bounding_box import compute_iou

class IoULoss(tf.keras.losses.Loss):
    """Implements the IoU Loss

    IoU loss is commonly used for object detection. This loss aims to directly
    optimize the IoU score between true boxes and predicted boxes. The length of the
    last dimension should be atleast 4 to represent the bounding boxes.
    
    Args:
        bounding_box_format: a case-insensitive string which is one of `"xyxy"`,
            `"rel_xyxy"`, `"xyWH"`, `"center_xyWH"`, `"yxyx"`, `"rel_yxyx"`. The
            position and shape of the bounding box will be followed by the class and
            confidence values (in that order). This is required for proper ranking of
            the bounding boxes. Therefore, each bounding box is defined by 6 values.
            For detailed information on the supported format, see the
            [KerasCV bounding box documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        mode: must be one of
            - `"linear"`. The loss will be calculated as 1 - iou
            - `"squared"`. The loss will be calculated as 1 - iou<sup>2</sup>
            - `"log"`. The loss will be calculated as -ln(iou)
            Defaults to "log".

    References:
        - [UnitBox paper](https://arxiv.org/pdf/1608.01471)

    Sample Usage:
    ```python
    y_true = tf.random.uniform([5, 10, 6], minval=0, maxval=10, dtype=tf.dtypes.int32)
    y_pred = tf.random.uniform([5, 10, 4], minval=0, maxval=10, dtype=tf.dtypes.int32)
    loss = IoULoss(bounding_box_format = "xyWH")
    loss(y_true, y_pred).numpy()
    ```
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.IoULoss())
    ```
    """

    def __init__(self, bounding_box_format, mode = 'log', **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.mode = mode

        if self.mode not in ['linear', 'square', 'log']:
            raise ValueError(
            "IoULoss expects mode to be one of 'linear', 'square' or 'log' "
            f"Received mode={self.mode}, "
        )

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        ious = compute_iou(y_true, y_pred, self.bounding_box_format)
        iou = tf.reduce_mean(ious, axis = [-2, -1])

        if self.mode == 'linear':
            loss = 1 - iou
        elif self.mode == 'square':
            loss = 1 - iou**2
        elif self.mode == 'log':
            loss = -tf.math.log(iou)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bounding_box_format": self.bounding_box_format,
                "mode": self.mode,
            }
        )
        return config
