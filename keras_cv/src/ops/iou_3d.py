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

from keras_cv.src.utils.resource_loader import LazySO

keras_cv_custom_ops = LazySO("custom_ops/_keras_cv_custom_ops.so")


def iou_3d(y_true, y_pred):
    """Implements IoU computation for 3D upright rotated bounding boxes.

    Note that this is implemented using a custom TensorFlow op. If you don't
    have KerasCV installed with custom ops, calling this will fail.

    Boxes should have the format CENTER_XYZ_DXDYDZ_PHI. Refer to
    https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box_3d/formats.py
    for more details on supported bounding box formats.

    Example:
    ```python
    y_true = [[0, 0, 0, 2, 2, 2, 0], [1, 1, 1, 2, 2, 2, 3 * math.pi / 4]]
    y_pred = [[1, 1, 1, 2, 2, 2, math.pi / 4], [1, 1, 1, 2, 2, 2, 0]]
    iou_3d(y_true, y_pred)
    ```
    """

    return keras_cv_custom_ops.ops.kcv_pairwise_iou3d(y_true, y_pred)
