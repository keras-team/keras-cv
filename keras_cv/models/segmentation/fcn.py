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

import tensorflow as tf

from keras_cv.models.vgg16 import VGG16
from keras_cv.models.weights import parse_weights


def get_vgg_layers(upsampling_factor):
    vgg16 = VGG16(include_rescaling=False, include_top='False', input_shape=(224,224,3))
    if upsampling_factor==32:
        return vgg16.get_layer('block5_pool').output
    elif upsampling_factor==16:
        return vgg16.get_layer('block4_pool').output
    elif upsampling_factor==8:
        return vgg16.get_layer('block3_pool').output
    else:
        raise ValueError(
            "The `upsampling_factor` should be either 32, 16 or 8"
        )

    