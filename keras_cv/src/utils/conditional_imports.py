# Copyright 2023 The KerasCV Authors
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

try:
    import waymo_open_dataset
except ImportError:
    waymo_open_dataset = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import matplotlib
except ImportError:
    matplotlib = None


try:
    import pycocotools
except ImportError:
    pycocotools = None


def assert_cv2_installed(symbol_name):
    if cv2 is None:
        raise ImportError(
            f"{symbol_name} requires the `cv2` package. "
            "Please install the package using "
            "`pip install opencv-python`."
        )


def assert_matplotlib_installed(symbol_name):
    if matplotlib is None:
        raise ImportError(
            f"{symbol_name} requires the `matplotlib` package. "
            "Please install the package using "
            "`pip install matplotlib`."
        )


def assert_waymo_open_dataset_installed(symbol_name):
    if waymo_open_dataset is None:
        raise ImportError(
            f"{symbol_name} requires the `waymo-open-dataset-tf` package. "
            "Please install the package from source. "
            "Installation instructions can be found at "
            "https://github.com/waymo-research/waymo-open-dataset"
            "/blob/master/docs/quick_start.md"
        )


def assert_pycocotools_installed(symbol_name):
    if pycocotools is None:
        raise ImportError(
            f"{symbol_name} requires the `pycocotools` package. "
            "Please install the package using "
            "`pip install pycocotools`."
        )
