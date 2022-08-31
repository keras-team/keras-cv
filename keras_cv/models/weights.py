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
    if weights in ALIASES[model_type]:
        weights = ALIASES[model_type][weights]
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

ALIASES = {
    "densenet121": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet169": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "resnet50v2": {
        "imagenet": "imagenet/classification-v1",
        "imagenet/classification": "imagenet/classification-v1",
    },
}

WEIGHTS_CONFIG = {
    "densenet121": {
        "imagenet/classification-v0": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",
        "imagenet/classification-v0-notop": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",
    },
    "densenet169": {
        "imagenet/classification-v0": "4cd2a661d0cb2378574073b23129ee4d06ea53c895c62a8863c44ee039e236a1",
        "imagenet/classification-v0-notop": "a99d1bb2cbe1a59a1cdd1f435fb265453a97c2a7b723d26f4ebee96e5fb49d62",
    },
    "densenet201": {},
    "resnet50v2": {
        "imagenet/classification-v0": "11bde945b54d1dca65101be2648048abca8a96a51a42820d87403486389790db",
        "imagenet/classification-v0-notop": "5b4aca4932c433d84f6aef58135472a4312ed2fa565d53fedcd6b0c24b54ab4a",
        "imagenet/classification-v1": "a32e5d9998e061527f6f947f36d8e794ad54dad71edcd8921cda7804912f3ee7",
        "imagenet/classification-v1-notop": "ac46b82c11070ab2f69673c41fbe5039c9eb686cca4f34cd1d79412fd136f1ae",
    },
}
