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

from keras_cv.bounding_box.converters import transform_format
from keras_cv.bounding_box.pad_batch_to_shape import pad_batch_to_shape

# per format axis selector constants


class XYXY:
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    CLASS = 4
    CONFIDENCE = 5


class CENTER_XYWH:
    X = 0
    Y = 1
    WIDTH = 2
    HEIGHT = 3
    CLASS = 4
    CONFIDENCE = 5


class XYWH:
    X = 0
    Y = 1
    WIDTH = 2
    HEIGHT = 3
    CLASS = 4
    CONFIDENCE = 5
