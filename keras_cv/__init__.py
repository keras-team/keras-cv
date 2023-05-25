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

# isort:off
import os

_USE_KERAS_CORE = False

if "USE_KERAS_CORE" in os.environ:
    _USE_KERAS_CORE = True


def use_keras_core():
    return _USE_KERAS_CORE


if use_keras_core():
    import keras_core

    register_keras_serializable = keras_core.saving.register_keras_serializable
else:
    from tensorflow import keras

    register_keras_serializable = keras.utils.register_keras_serializable

from keras_cv import version_check

version_check.check_tf_version()
# isort:on

from keras_cv import bounding_box
from keras_cv import callbacks
from keras_cv import datasets
from keras_cv import layers
from keras_cv import losses
from keras_cv import metrics
from keras_cv import models
from keras_cv import training
from keras_cv import utils
from keras_cv import visualization
from keras_cv.core import ConstantFactorSampler
from keras_cv.core import FactorSampler
from keras_cv.core import NormalFactorSampler
from keras_cv.core import UniformFactorSampler

__version__ = "0.5.0"
