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


class CENTER_XYZ_DXDYDZ_PHI:
    """CENTER_XYZ_DXDYDZ_PHI contains axis indices for the CENTER_XYZ_DXDYDZ_PHI
    format.

    CENTER_XYZ_DXDYDZ_PHI is a 3D box format that supports vertical boxes with a
    heading rotated around the Z axis.

    The CENTER_XYZ_DXDYDZ_PHI format consists of the following required indices:

    - X: X coordinate of the center of the bounding box
    - Y: Y coordinate of the center of the bounding box
    - Z: Z coordinate of the center of the bounding box
    - DX: size of the bounding box on the x-axis
    - DY: size of the bounding box on the y-axis
    - DZ: size of the bounding box on the z-axis
    - PHI: the rotation of the box with respect to the z axis, in radians

    and the following optional indices, used in some KerasCV components:

    - CLASS: class of the object contained in the bounding box
    """

    X = 0
    Y = 1
    Z = 2
    DX = 3
    DY = 4
    DZ = 5
    PHI = 6
    CLASS = 7
