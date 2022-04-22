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

"""KerasCV Version check."""


import warnings

import tensorflow as tf
from packaging.version import parse

MIN_VERSION = "2.9.0"


def check_tf_version():
    if parse(tf.__version__) < parse(MIN_VERSION):
        warnings.warn(
            f"The Tensorflow package version needs to be at least {MIN_VERSION} \n"
            "for KerasCV to run. Currently, your TensorFlow version is \n"
            f"{tf.__version__}. Please upgrade with \n"
            "`$ pip install --upgrade tensorflow`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.",
            ImportWarning,
        )
