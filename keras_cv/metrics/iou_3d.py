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
"""IoU3D metric using a custom TF op."""

import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


class IoU3D(tf.keras.metrics.MeanMetricWrapper):
    """Implements the IoU metric for 3D upright rotated bounding boxes.

    Note that this is implemented using a custom TensorFlow op. Initializing an
    IoU3D object will attempt to load the binary for that op.

    Boxes should have the format [center_x, center_y, center_z, dimension_x,
    dimension_y, dimension_z, heading (in radians)].

    Sample Usage:
    ```python
    y_true = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
    y_pred = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]
    metric = IoU3D()
    metric.update_state(y_true, y_pred)
    metric.result()
    ```
    """

    def __init__(self, name="IoU3D", **kwargs):
        pairwise_iou_op = load_library.load_op_library(
            resource_loader.get_path_to_datafile("../custom_ops/_pairwise_iou_op.so")
        )
        iou_3d = pairwise_iou_op.pairwise_iou3d

        super().__init__(iou_3d, name=name, **kwargs)
