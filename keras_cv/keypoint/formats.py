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


class XY:
    """XY contains axis indices for the XY format.

    All values in the XY format should be absolute pixel values.

    The XY format consists of the following required indices:

    - X: the width position
    - Y: the height position

    and the following optional indices, used in some KerasCV components:

    - CLASS: class of the keypoints
    - CONFIDENCE: confidence of the keypoints
    """

    X = 0
    Y = 1
    CLASS = 2
    CONFIDENCE = 3


class REL_XY:
    """REL_XY contains axis indices for the REL_XY format.


    REL_XY is like XY, but each value is relative to the width and height of the
    origin image.  Values are percentages of the origin images' width and height
    respectively.

    The REL_XY format consists of the following required indices:

    - X: the width position
    - Y: the height position

    and the following optional indices, used in some KerasCV components:

    - CLASS: class of the keypoints
    - CONFIDENCE: confidence of the keypoints
    """

    X = 0
    Y = 1
    CLASS = 2
    CONFIDENCE = 3
