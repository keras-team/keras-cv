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


import math
import tensorflow as tf
from tensorflow import keras


class CIoULoss(keras.losses.Loss):
    """Implements the Complete IoU (CIoU) Loss

    CIoU loss is an extension of GIoU loss, which further improves the IoU
    optimization for object detection. CIoU loss not only penalizes the
    bounding box coordinates but also considers the aspect ratio and center
    distance of the boxes. The length of the last dimension should be 4 to
    represent the bounding boxes.

    Args:
        bounding_box_format: a case-insensitive string (for example, "xyxy").
            Each bounding box is defined by these 4 values. For detailed
            information on the supported formats, see the [KerasCV bounding box
            documentation](https://keras.io/api/keras_cv/bounding_box/formats/).
        axis: the axis along which to mean the ious, defaults to -1.

    References:
        - [CIoU paper](https://arxiv.org/pdf/2005.03572.pdf)
        - [TFAddons Implementation](https://www.tensorflow.org/addons/api_docs/python/tfa/losses/CIoULoss)

    Sample Usage:
    ```python
    y_true = tf.random.uniform(
        (5, 10, 5),
        minval=0,
        maxval=10,
        dtype=tf.dtypes.int32)
    y_pred = tf.random.uniform(
        (5, 10, 4),
        minval=0,
        maxval=10,
        dtype=tf.dtypes.int32)
    loss = CIoULoss(bounding_box_format="xyWH")
    loss(y_true, y_pred).numpy()
    ```

    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=CIoULoss())
    ```
    """
    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def compute_ciou(self, boxes1, boxes2):
        b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(boxes1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(boxes2, 4, axis=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps

        # Intersection area
        inter = tf.math.maximum(
            tf.math.minimum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1), 0
        ) * tf.math.maximum(
            tf.math.minimum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1), 0
        )

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + self.eps

        # IoU
        iou = inter / union

        cw = tf.math.maximum(b1_x2, b2_x2) - tf.math.minimum(
        b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = tf.math.maximum(b1_y2, b2_y2) - tf.math.minimum(
            b1_y1, b2_y1
        )  # convex height
        c2 = cw**2 + ch**2 + self.eps  # convex diagonal squared
        rho2 = (
            (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
        ) / 4  # center dist ** 2
        v = tf.pow((4 / math.pi**2) * (tf.atan(w2 / h2) - tf.atan(w1 / h1)), 2)
        alpha = v / (v - iou + (1 + self.eps))
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou
    

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if y_pred.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_pred.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_pred.shape[-1]={y_pred.shape[-1]}."
            )

        if y_true.shape[-1] != 4:
            raise ValueError(
                "CIoULoss expects y_true.shape[-1] to be 4 to represent the "
                f"bounding boxes. Received y_true.shape[-1]={y_true.shape[-1]}."
            )

        if y_true.shape[-2] != y_pred.shape[-2]:
            raise ValueError(
                "CIoULoss expects number of boxes in y_pred to be equal to the "
                "number of boxes in y_true. Received number of boxes in "
                f"y_true={y_true.shape[-2]} and number of boxes in "
                f"y_pred={y_pred.shape[-2]}."
            )

        ciou = self.compute_ciou(y_true, y_pred)

        return 1 - ciou

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
            }
        )
        return config
