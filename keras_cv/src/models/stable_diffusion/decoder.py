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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models.stable_diffusion.attention_block import (  # noqa: E501
    AttentionBlock,
)
from keras_cv.src.models.stable_diffusion.padded_conv2d import PaddedConv2D
from keras_cv.src.models.stable_diffusion.resnet_block import ResnetBlock


@keras_cv_export("keras_cv.models.stable_diffusion.Decoder")
class Decoder(keras.Sequential):
    def __init__(self, img_height, img_width, name=None, download_weights=True):
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
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(3, 3, padding=1),
            ],
            name=name,
        )

        if download_weights:
            decoder_weights_fpath = keras.utils.get_file(
                origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/kcv_decoder.h5",  # noqa: E501
                file_hash="ad350a65cc8bc4a80c8103367e039a3329b4231c2469a1093869a345f55b1962",  # noqa: E501
            )
            self.load_weights(decoder_weights_fpath)
