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
    "darknet53": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet121": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet169": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "densenet201": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "efficientnetv2b0": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "efficientnetv2b1": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "efficientnetv2b2": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "efficientnetv2s": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "resnet50v2": {
        "imagenet": "imagenet/classification-v2",
        "imagenet/classification": "imagenet/classification-v2",
    },
}

WEIGHTS_CONFIG = {
    "darknet53": {
        "imagenet/classification-v0": "7bc5589f7f7f7ee3878e61ab9323a71682bfb617eb57f530ca8757c742f00c77",
        "imagenet/classification-v0-notop": "8dcce43163e4b4a63e74330ba1902e520211db72d895b0b090b6bfe103e7a8a5",
    },
    "densenet121": {
        "imagenet/classification-v0": "13de3d077ad9d9816b9a0acc78215201d9b6e216c7ed8e71d69cc914f8f0775b",
        "imagenet/classification-v0-notop": "709afe0321d9f2b2562e562ff9d0dc44cca10ed09e0e2cfba08d783ff4dab6bf",
    },
    "densenet169": {
        "imagenet/classification-v0": "4cd2a661d0cb2378574073b23129ee4d06ea53c895c62a8863c44ee039e236a1",
        "imagenet/classification-v0-notop": "a99d1bb2cbe1a59a1cdd1f435fb265453a97c2a7b723d26f4ebee96e5fb49d62",
    },
    "densenet201": {
        "imagenet/classification-v0": "3b6032e744e5e5babf7457abceaaba11fcd449fe2d07016ae5076ac3c3c6cf0c",
        "imagenet/classification-v0-notop": "c1189a934f12c1a676a9cf52238e5994401af925e2adfc0365bad8133c052060",
    },
    "efficientnetv2b0": {
        "imagenet/classification-v0": "da7975b6d4200dfdc3f859b0d028774e5e5dd4031d3e998a27dadc492dec4f3e",
        "imagenet/classification-v0-notop": "defe635bfa3cc3f2b9e89bfd53bbc3de28a1dc67026b4437a14f44476e7d0549",
    },
    "efficientnetv2b1": {
        "imagenet/classification-v0": "3f92fc9d7b141ec9e85ffe60d301fb49103ba17b148bdd638971a77f1b8db010",
        "imagenet/classification-v0-notop": "359aaa5c1e863c8438d94052791e72ef29345d07703d06284e1069829f85932f",
    },
    "efficientnetv2b2": {
        "imagenet/classification-v0": "1667d21b50e6c5b851a69c98503fa5ae707b82dbae8c900fe59ab1a93d60d694",
        "imagenet/classification-v0-notop": "e118aadfab7e93ff939fb81c88c189cbd7fb2b7ddd7314fbf2badb7c551aa119",
    },
    "efficientnetv2s": {
        "imagenet/classification-v0": "77c8fb0ea9cbf6277c6c3cdfece00e610aa1d19edf3d76d15b4e6bcbbeada904",
        "imagenet/classification-v0-notop": "9652f71d398c2de6c595c68b881f257d715bcf0cdc5adb80e95ce83e828ae2c6",
    },
    "resnet50v2": {
        "imagenet/classification-v0": "11bde945b54d1dca65101be2648048abca8a96a51a42820d87403486389790db",
        "imagenet/classification-v0-notop": "5b4aca4932c433d84f6aef58135472a4312ed2fa565d53fedcd6b0c24b54ab4a",
        "imagenet/classification-v1": "a32e5d9998e061527f6f947f36d8e794ad54dad71edcd8921cda7804912f3ee7",
        "imagenet/classification-v1-notop": "ac46b82c11070ab2f69673c41fbe5039c9eb686cca4f34cd1d79412fd136f1ae",
        "imagenet/classification-v2": "5ee5a8ac650aaa59342bc48ffe770e6797a5550bcc35961e1d06685292c15921",
        "imagenet/classification-v2-notop": "e711c83d6db7034871f6d345a476c8184eab99dbf3ffcec0c1d8445684890ad9",
    },
}
