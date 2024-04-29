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
"""
formats.py contains axis information for each supported format.
"""

from keras_cv.src.api_export import keras_cv_export


@keras_cv_export("keras_cv.bounding_box.XYXY")
class XYXY:
    """XYXY contains axis indices for the XYXY format.

    All values in the XYXY format should be absolute pixel values.

    The XYXY format consists of the following required indices:

    - LEFT: left of the bounding box
    - TOP: top of the bounding box
    - RIGHT: right of the bounding box
    - BOTTOM: bottom of the bounding box
    """

    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3


@keras_cv_export("keras_cv.bounding_box.REL_XYXY")
class REL_XYXY:
    """REL_XYXY contains axis indices for the REL_XYXY format.

    REL_XYXY is like XYXY, but each value is relative to the width and height of
    the origin image. Values are percentages of the origin images' width and
    height respectively.

    The REL_XYXY format consists of the following required indices:

    - LEFT: left of the bounding box
    - TOP: top of the bounding box
    - RIGHT: right of the bounding box
    - BOTTOM: bottom of the bounding box
    """

    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3


@keras_cv_export("keras_cv.bounding_box.CENTER_XYWH")
class CENTER_XYWH:
    """CENTER_XYWH contains axis indices for the CENTER_XYWH format.

    All values in the CENTER_XYWH format should be absolute pixel values.

    The CENTER_XYWH format consists of the following required indices:

    - X: X coordinate of the center of the bounding box
    - Y: Y coordinate of the center of the bounding box
    - WIDTH: width of the bounding box
    - HEIGHT: height of the bounding box
    """

    X = 0
    Y = 1
    WIDTH = 2
    HEIGHT = 3


@keras_cv_export("keras_cv.bounding_box.XYWH")
class XYWH:
    """XYWH contains axis indices for the XYWH format.

    All values in the XYWH format should be absolute pixel values.

    The XYWH format consists of the following required indices:

    - X: X coordinate of the left of the bounding box
    - Y: Y coordinate of the top of the bounding box
    - WIDTH: width of the bounding box
    - HEIGHT: height of the bounding box
    """

    X = 0
    Y = 1
    WIDTH = 2
    HEIGHT = 3


@keras_cv_export("keras_cv.bounding_box.REL_XYWH")
class REL_XYWH:
    """REL_XYWH contains axis indices for the XYWH format.

    REL_XYXY is like XYWH, but each value is relative to the width and height of
    the origin image. Values are percentages of the origin images' width and
    height respectively.

    - X: X coordinate of the left of the bounding box
    - Y: Y coordinate of the top of the bounding box
    - WIDTH: width of the bounding box
    - HEIGHT: height of the bounding box
    """

    X = 0
    Y = 1
    WIDTH = 2
    HEIGHT = 3


@keras_cv_export("keras_cv.bounding_box.YXYX")
class YXYX:
    """YXYX contains axis indices for the YXYX format.

    All values in the YXYX format should be absolute pixel values.

    The YXYX format consists of the following required indices:

    - TOP: top of the bounding box
    - LEFT: left of the bounding box
    - BOTTOM: bottom of the bounding box
    - RIGHT: right of the bounding box
    """

    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3


@keras_cv_export("keras_cv.bounding_box.REL_YXYX")
class REL_YXYX:
    """REL_YXYX contains axis indices for the REL_YXYX format.

    REL_YXYX is like YXYX, but each value is relative to the width and height of
    the origin image. Values are percentages of the origin images' width and
    height respectively.

    The REL_YXYX format consists of the following required indices:

    - TOP: top of the bounding box
    - LEFT: left of the bounding box
    - BOTTOM: bottom of the bounding box
    - RIGHT: right of the bounding box
    """

    TOP = 0
    LEFT = 1
    BOTTOM = 2
    RIGHT = 3
