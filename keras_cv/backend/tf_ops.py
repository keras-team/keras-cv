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
from keras_cv.backend import config

if config.multi_backend():
    try:
        from tensorflow.keras.backend.tensorflow import *  # noqa: F403, F401
        from tensorflow.keras.backend.tensorflow import (
            convert_to_numpy,
        )  # noqa: F403, F401
        from tensorflow.keras.backend.tensorflow.core import *  # noqa: F403, F401
        from tensorflow.keras.backend.tensorflow.math import *  # noqa: F403, F401
        from tensorflow.keras.backend.tensorflow.nn import *  # noqa: F403, F401
        from tensorflow.keras.backend.tensorflow.numpy import *  # noqa: F403, F401
    except ImportError:
        from keras_core.src.backend.tensorflow import *  # noqa: F403, F401
        from keras_core.src.backend.tensorflow import (
            convert_to_numpy,
        )  # noqa: F403, F401
        from keras_core.src.backend.tensorflow.core import *  # noqa: F403, F401
        from keras_core.src.backend.tensorflow.math import *  # noqa: F403, F401
        from keras_core.src.backend.tensorflow.nn import *  # noqa: F403, F401
        from keras_core.src.backend.tensorflow.numpy import *  # noqa: F403, F401 , E501


# Some TF APIs where the numpy API doesn't support raggeds that we need
from tensorflow import broadcast_to  # noqa: F403, F401
from tensorflow import repeat  # noqa: F403, F401
from tensorflow import reshape  # noqa: F403, F401
from tensorflow import split  # noqa: F403, F401
from tensorflow import concat as concatenate  # noqa: F403, F401
from tensorflow import range as arange  # noqa: F403, F401
from tensorflow import reduce_all as all  # noqa: F403, F401
from tensorflow import reduce_max as max  # noqa: F403, F401
from tensorflow.keras.preprocessing.image import (
    smart_resize,
)  # noqa: F403, F401
