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
from keras_cv.backend.config import backend
from keras_cv.backend.config import detect_if_tensorflow_uses_keras_3

if detect_if_tensorflow_uses_keras_3():
    from tensorflow.keras.backend import vectorized_map  # noqa: F403, F401
    from tensorflow.keras.ops import *  # noqa: F403, F401
    from tensorflow.keras.utils.image_utils import (  # noqa: F403, F401
        smart_resize,
    )
else:
    try:
        from keras.src.backend import vectorized_map  # noqa: F403, F401
        from keras.src.ops import *  # noqa: F403, F401
        from keras.src.utils.image_utils import smart_resize  # noqa: F403, F401
    # Import error means Keras isn't installed, or is Keras 2.
    except ImportError:
        from keras_core.src.backend import vectorized_map  # noqa: F403, F401
        from keras_core.src.ops import *  # noqa: F403, F401
        from keras_core.src.utils.image_utils import (  # noqa: F403, F401
            smart_resize,
        )
if backend() == "Tensorflow":
    from keras_cv.backend.tf_ops import *  # noqa: F403, F401
