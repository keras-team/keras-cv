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
from keras.utils import data_utils


def parse_weights(weights, include_top, model_type):
    if not a or tf.io.gfile.exists(weights):
        return weights
    if weights in WEIGHTS_CONFIG[model_type]:
        if not include_top:
            weights = weights + '-notop'
        return data_utils.get_file(
            f"{BASE_PATH}/{model_type}/{weights}.h5",
            cache_subdir="models",
            file_hash=WEIGHTS_CONFIG[model_type][weights],
        )

    raise ValueError(
        "Invalid weights parameter. Must be either `None`, a path to local weights, or a supported pre-trained weights name. See weights.py for a list of supported pre-trained weights"
    )

BASE_PATH = "https://storage.googleapis.com/keras-cv/models"
WEIGHTS_CONFIG = {
    "densenet121": {
        # Current best weights are imagenet-v0
        "imagenet": "5c51af007f7f3722b50d9390db4b2082962d5ba1ab5d184c3f531f3886200894",
        "imagenet-notop": "5c51af007f7f3722b50d9390db4b2082962d5ba1ab5d184c3f531f3886200894",
        "imagenet-v0": "5c51af007f7f3722b50d9390db4b2082962d5ba1ab5d184c3f531f3886200894",
        "imagenet-v0-notop": "5c51af007f7f3722b50d9390db4b2082962d5ba1ab5d184c3f531f3886200894",

    },
    "densenet169": {},
    "densenet201": {},
}
