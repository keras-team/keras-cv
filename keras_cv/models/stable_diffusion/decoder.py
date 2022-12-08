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

from tensorflow import keras

import keras_cv.models.stable_diffusion.weights as weights_lib
from keras_cv.models.stable_diffusion.__internal__.layers.attention_block import (
    AttentionBlock,
)
from keras_cv.models.stable_diffusion.__internal__.layers.group_normalization import (
    GroupNormalization,
)
from keras_cv.models.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)

preconfigured_weights = {
    "v1": "https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",
    "v1.5": "https://huggingface.co/Lukewood/sd-1.5-keras-cv-weights/resolve/main/decoder.h5",
}

hashes = {
    "https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5": "ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",
    "https://huggingface.co/Lukewood/sd-1.5-keras-cv-weights/resolve/main/decoder.h5": "22ef0208c1ea3495febc440876f5b533bdb08f288108b743b64977c5bba50882",
}


class Decoder(keras.Sequential):
    def __init__(self, img_height, img_width, name=None, weights="v1"):
        super().__init__(
            [
                keras.layers.Input((img_height // 8, img_width // 8, 4)),
                keras.layers.Rescaling(1.0 / 0.18215),
                PaddedConv2D(4, 1),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(512, 3, padding=1),
                ResnetBlock(256),
                ResnetBlock(256),
                ResnetBlock(256),
                keras.layers.UpSampling2D(2),
                PaddedConv2D(256, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                ResnetBlock(128),
                GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ],
            name=name,
        )
        weights_lib.load_weights(self, weights, preconfigured_weights, hashes)
