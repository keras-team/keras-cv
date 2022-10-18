# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""IoU3D loss using a custom TF op."""

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


class IoU3DLoss(tf.keras.losses.Loss):
    """Implements the IoU Loss for 3D upright rotated bounding boxes.

    Note that this is implemented using a custom TensorFlow op. Initializing an
    IoU3DLoss object will attempt to load the binary for that op.

    IoU loss is commonly used for object detection. This loss aims to directly
    optimize the IoU score between true boxes and predicted boxes. Boxes should
    have the format [center_x, center_y, center_z, dimension_x, dimension_y,
    dimension_y, heading (in radians)].

    Sample Usage:
    ```python
    y_true = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
    y_pred = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]
    loss = IoU3DLoss()
    loss(y_true, y_pred)
    ```
    Usage with the `compile()` API:
    ```python
    model.compile(optimizer='adam', loss=keras_cv.losses.IoU3DLoss())
    ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        pairwise_iou_op = load_library.load_op_library(
            resource_loader.get_path_to_datafile("../custom_ops/_pairwise_iou_op.so")
        )
        self.iou_3d = pairwise_iou_op.pairwise_iou3d

    def call(self, y_true, y_pred):
        return self.iou_3d(y_true, y_pred)
