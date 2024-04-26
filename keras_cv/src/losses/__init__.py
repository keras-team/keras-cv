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

from keras_cv.src.losses.centernet_box_loss import CenterNetBoxLoss
from keras_cv.src.losses.ciou_loss import CIoULoss
from keras_cv.src.losses.focal import FocalLoss
from keras_cv.src.losses.giou_loss import GIoULoss
from keras_cv.src.losses.iou_loss import IoULoss
from keras_cv.src.losses.penalty_reduced_focal_loss import (
    BinaryPenaltyReducedFocalCrossEntropy,
)
from keras_cv.src.losses.simclr_loss import SimCLRLoss
from keras_cv.src.losses.smooth_l1 import SmoothL1Loss
