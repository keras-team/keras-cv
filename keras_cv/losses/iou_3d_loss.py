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
"""IoU3DLoss in python using a custom TF op."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

iou_3d_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile("../custom_ops/_zero_out_ops.so")
)
iou_3d = iou_3d_ops.zero_out
