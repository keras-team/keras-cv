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
from tensorflow.keras import layers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss
from keras_cv.training import ContrastiveTrainer


class SimCLRTrainer(ContrastiveTrainer):
    """Creates a SimCLRTrainer.

    References:
        - [SimCLR paper](https://arxiv.org/pdf/2002.05709)

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        include_probe: Whether to include a single fully-connected layer during
            training for probing classification accuracy using the learned encoding.
            Note that this should be specified iff training with labeled images.
            If provided, `classes` must be provided.
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is setup.
        target_size: A tuple of two integers used as the target size to resize
            images to in the augmentation setp.
    """

    def __init__(
        self,
        encoder,
        include_probe,
        value_range,
        target_size=(128, 128),
        projection_width=128,
        **kwargs
    ):
        super().__init__(
            encoder=encoder,
            augmenter=preprocessing.Augmenter(
                [
                    preprocessing.RandomFlip(),
                    preprocessing.RandomResizedCrop(
                        target_size,
                        crop_area_factor=(0.08, 1),
                        aspect_ratio_factor=(3 / 4, 4 / 3),
                    ),
                    preprocessing.RandomColorJitter(
                        value_range=value_range,
                        brightness_factor=0.25,
                        contrast_factor=0.5,
                        saturation_factor=(0.3, 0.7),
                        hue_factor=0.2,
                    ),
                ]
            ),
            projector=keras.Sequential(
                [
                    layers.Dense(projection_width, activation="relu"),
                    layers.Dense(projection_width),
                ],
                name="projector",
            ),
            include_probe=include_probe,
            **kwargs,
        )
