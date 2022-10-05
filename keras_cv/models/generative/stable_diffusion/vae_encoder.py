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

from keras_cv.models.generative.stable_diffusion.__internal__.layers.attention_block import (
    AttentionBlock,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.group_normalization import (
    GroupNormalization,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.padded_conv2d import (
    PaddedConv2D,
)
from keras_cv.models.generative.stable_diffusion.__internal__.layers.resnet_block import (
    ResnetBlock,
)


class VAEEncoder(keras.Sequential):
    def __init__(self):
        super().__init__(
            [
                keras.layers.Input((512, 512, 3)),
                PaddedConv2D(128, 3, padding=1),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv2D(128, 3, padding=1, strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv2D(256, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv2D(512, 3, padding=1, strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv2D(8, 3, padding=1),
                PaddedConv2D(8, 1),
                # TODO(lukewood): can this be refactored to be a Rescaling layer?
                # Perhaps some sort of rescale and gather?
                # Either way, we may need a lambda to gather the first 4 dimensions.
                keras.layers.Lambda(lambda x: x[..., :4] * 0.18215),
            ]
        )
