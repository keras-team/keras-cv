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
    "cspdarknet": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "darknet53": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "deeplabv3": {
        "voc": "voc/segmentation-v0",
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
    "resnet50": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "resnet50v2": {
        "imagenet": "imagenet/classification-v2",
        "imagenet/classification": "imagenet/classification-v2",
    },
    "vittiny16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vits16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitb16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitl16": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vits32": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
    "vitb32": {
        "imagenet": "imagenet/classification-v0",
        "imagenet/classification": "imagenet/classification-v0",
    },
}

WEIGHTS_CONFIG = {
    "cspdarknet": {
        "imagenet/classification-v0": "8bdc3359222f0d26f77aa42c4e97d67a05a1431fe6c448ceeab9a9c5a34ff804",
        "imagenet/classification-v0-notop": "9303aabfadffbff8447171fce1e941f96d230d8f3cef30d3f05a9c85097f8f1e",
    },
    "darknet53": {
        "imagenet/classification-v0": "7bc5589f7f7f7ee3878e61ab9323a71682bfb617eb57f530ca8757c742f00c77",
        "imagenet/classification-v0-notop": "8dcce43163e4b4a63e74330ba1902e520211db72d895b0b090b6bfe103e7a8a5",
    },
    "deeplabv3": {
        "voc/segmentation-v0": "732042e8b6c9ddba3d51c861f26dc41865187e9f85a0e5d43dfef75a405cca18",
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
        "imagenet/classification-v0": "dbde38e7c56af5bdafe61fd798cf5d490f3c5e3b699da7e25522bc828d208984",
        "imagenet/classification-v0-notop": "ac95f13a8ad1cee41184fc16fd0eb769f7c5b3131151c6abf7fcee5cc3d09bc8",
    },
    "efficientnetv2b1": {
        "imagenet/classification-v0": "9dd8f3c8de3bbcc269a1b9aed742bb89d56be445b6aa271aa6037644f4210e9a",
        "imagenet/classification-v0-notop": "82da111f8411f47e3f5eef090da76340f38e222f90a08bead53662f2ebafb01c",
    },
    "efficientnetv2b2": {
        "imagenet/classification-v0": "05eb5674e0ecbf34d5471f611bcfa5da0bb178332dc4460c7a911d68f9a2fe87",
        "imagenet/classification-v0-notop": "02d12c9d1589b540b4e84ffdb54ff30c96099bd59e311a85ddc7180efc65e955",
    },
    "efficientnetv2s": {
        "imagenet/classification-v0": "2259db3483a577b5473dd406d1278439bd1a704ee477ff01a118299b134bd4db",
        "imagenet/classification-v0-notop": "80555436ea49100893552614b4dce98de461fa3b6c14f8132673817d28c83654",
    },
    "resnet50": {
        "imagenet/classification-v0": "1525dc1ce580239839ba6848c0f1b674dc89cb9ed73c4ed49eba355b35eac3ce",
        "imagenet/classification-v0-notop": "dc5f6d8f929c78d0fc192afecc67b11ac2166e9d8b9ef945742368ae254c07af",
    },
    "resnet50v2": {
        "imagenet/classification-v0": "11bde945b54d1dca65101be2648048abca8a96a51a42820d87403486389790db",
        "imagenet/classification-v0-notop": "5b4aca4932c433d84f6aef58135472a4312ed2fa565d53fedcd6b0c24b54ab4a",
        "imagenet/classification-v1": "a32e5d9998e061527f6f947f36d8e794ad54dad71edcd8921cda7804912f3ee7",
        "imagenet/classification-v1-notop": "ac46b82c11070ab2f69673c41fbe5039c9eb686cca4f34cd1d79412fd136f1ae",
        "imagenet/classification-v2": "5ee5a8ac650aaa59342bc48ffe770e6797a5550bcc35961e1d06685292c15921",
        "imagenet/classification-v2-notop": "e711c83d6db7034871f6d345a476c8184eab99dbf3ffcec0c1d8445684890ad9",
    },
    "vittiny16": {
        "imagenet/classification-v0": "c8227fde16ec8c2e7ab886169b11b4f0ca9af2696df6d16767db20acc9f6e0dd",
        "imagenet/classification-v0-notop": "aa4d727e3c6bd30b20f49d3fa294fb4bbef97365c7dcb5cee9c527e4e83c8f5b",
    },
    "vits16": {
        "imagenet/classification-v0": "4a66a1a70a879ff33a3ca6ca30633b9eadafea84b421c92174557eee83e088b5",
        "imagenet/classification-v0-notop": "8d0111eda6692096676a5453abfec5d04c79e2de184b04627b295f10b1949745",
    },
    "vitb16": {
        "imagenet/classification-v0": "6ab4e08c773e08de42023d963a97e905ccba710e2c05ef60c0971978d4a8c41b",
        "imagenet/classification-v0-notop": "4a1bdd32889298471cb4f30882632e5744fd519bf1a1525b1fa312fe4ea775ed",
    },
    "vitl16": {
        "imagenet/classification-v0": "5a98000f848f2e813ea896b2528983d8d956f8c4b76ceed0b656219d5b34f7fb",
        "imagenet/classification-v0-notop": "40d237c44f14d20337266fce6192c00c2f9b890a463fd7f4cb17e8e35b3f5448",
    },
    "vits32": {
        "imagenet/classification-v0": "f5836e3aff2bab202eaee01d98337a08258159d3b718e0421834e98b3665e10a",
        "imagenet/classification-v0-notop": "f3907845eff780a4d29c1c56e0ae053411f02fff6fdce1147c4c3bb2124698cd",
    },
    "vitb32": {
        "imagenet/classification-v0": "73025caa78459dc8f9b1de7b58f1d64e24a823f170d17e25fcc8eb6179bea179",
        "imagenet/classification-v0-notop": "f07b80c03336d731a2a3a02af5cac1e9fc9aa62659cd29e2e7e5c7474150cc71",
    },
}
