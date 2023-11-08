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
from keras_cv.backend import config  # noqa: E402
from keras_cv.backend import keras  # noqa: E402
from keras_cv.backend import ops  # noqa: E402
from keras_cv.backend import random  # noqa: E402
from keras_cv.backend import tf_ops  # noqa: E402


def assert_tf_keras(src):
    if config.multi_backend():
        raise NotImplementedError(
            f"KerasCV component {src} does not yet support Keras Core, and can "
            "only be used in `tf.keras`."
        )


def supports_ragged():
    return not config.multi_backend()
