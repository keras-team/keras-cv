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
import tensorflow as tf
from keras.utils import data_utils


def parse_weights(weights, include_top, model_type):
    if not weights or tf.io.gfile.exists(weights):
        return weights
    if weights in WEIGHTS_CONFIG[model_type]:
        if not include_top:
            weights = weights + "-notop"
        return data_utils.get_file(
            origin=f"{BASE_PATH}/{model_type}/{weights}.h5",
            cache_subdir="models",
            file_hash=WEIGHTS_CONFIG[model_type][weights],
        )

    raise ValueError(
        "The `weights` argument should be either `None`, a the path to the "
        "weights file to be loaded, or the name of pre-trained weights from "
        "https://github.com/keras-team/keras-cv/blob/master/keras_cv/models/weights.py. "
        f"Invalid `weights` argument: {weights}"
    )


BASE_PATH = "https://storage.googleapis.com/keras-cv/models"
WEIGHTS_CONFIG = {
    "densenet121": {
        # Current best weights are imagenet-v0
        "imagenet/classification": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",
        "imagenet/classification-notop": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",
        "imagenet/classification-v0": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",
        "imagenet/classification-v0-notop": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",
    },
    "densenet169": {},
    "densenet201": {},
}
