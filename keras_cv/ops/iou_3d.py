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
"""IoU3D using a custom TF op."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

iou_3d = None
try:
    pairwise_iou_op = load_library.load_op_library(
        resource_loader.get_path_to_datafile("../custom_ops/_keras_cv_custom_ops.so")
    )
    iou_3d = pairwise_iou_op.pairwise_iou3d
except:
    print("Loading KerasCV without custom ops.")


class IoU3D:
    """Implements IoU computation for 3D upright rotated bounding boxes.

    Note that this is implemented using a custom TensorFlow op. Initializing an
    IoU3D object will attempt to load the binary for that op.

    Boxes should have the format [center_x, center_y, center_z, dimension_x,
    dimension_y, dimension_z, heading (in radians)].

    Sample Usage:
    ```python
    y_true = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
    y_pred = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]
    iou = IoU3D()
    iou(y_true, y_pred)
    ```
    """

    def __init__(self):
        pairwise_iou_op = load_library.load_op_library(
            resource_loader.get_path_to_datafile(
                "../custom_ops/_keras_cv_custom_ops.so"
            )
        )
        self.iou_3d = iou_3d

    def __call__(self, y_true, y_pred):
        return self.iou_3d(y_true, y_pred)
