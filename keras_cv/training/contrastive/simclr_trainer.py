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

from keras_cv.layers import preprocessing


class SimCLRTrainer(preprocessing.Augmenter):
    """Creates a SimCLRTrainer.

    References:
        - [SimCLR paper](https://arxiv.org/pdf/2002.05709)

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
    """

    def __init__(self, encoder, value_range):
        super().__init__(
            encoder=encoder,
            augmentor=preprocessing.Augmenter(
                [
                    preprocessing.RandomFlip(),
                    preprocessing.RandomTranslation(0.25, 0.25),
                    preprocessing.RandomZoom((-0.5, 0.0), (-0.5, 0.0)),
                    preprocessing.RandomColorJitter(
                        value_range=value_range,
                        brightness_factor=0.5,
                        contrast_factor=0.5,
                        saturation_factor=(0.3, 0.7),
                        hue_factor=0.5,
                    ),
                ]
            ),
            projector=keras.Sequential(
                [
                    layers.Dense(self.projection_width, activation="relu"),
                    layers.Dense(self.projection_width),
                ],
                name="projector",
            ),
        )
