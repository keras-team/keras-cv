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
import copy
import functools

from keras_cv.src import backend
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.backend import tf_ops
from keras_cv.src.backend.config import keras_3

_ORIGINAL_OPS = copy.copy(backend.ops.__dict__)
_ORIGINAL_SUPPORTS_RAGGED = backend.supports_ragged

# A counter for potentially nested TF data scopes
_IN_TF_DATA_SCOPE = 0


def tf_data(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if keras_3() and keras.src.utils.backend_utils.in_tf_graph():
            with TFDataScope():
                return function(*args, **kwargs)
        else:
            return function(*args, **kwargs)

    return wrapper


class TFDataScope:
    def __enter__(self):
        global _IN_TF_DATA_SCOPE
        if _IN_TF_DATA_SCOPE == 0:
            for k, v in ops.__dict__.items():
                if k in tf_ops.__dict__:
                    setattr(ops, k, getattr(tf_ops, k))
            backend.supports_ragged = lambda: True
        _IN_TF_DATA_SCOPE += 1

    def __exit__(self, exc_type, exc_value, exc_tb):
        global _IN_TF_DATA_SCOPE
        _IN_TF_DATA_SCOPE -= 1
        if _IN_TF_DATA_SCOPE == 0:
            for k, v in ops.__dict__.items():
                setattr(ops, k, _ORIGINAL_OPS[k])
            backend.supports_ragged = _ORIGINAL_SUPPORTS_RAGGED
            _IN_TF_DATA_SCOPE = False
