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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.layers import preprocessing
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@keras_cv_export("keras_cv.layers.RandomAugmentationPipeline")
class RandomAugmentationPipeline(BaseImageAugmentationLayer):
    """RandomAugmentationPipeline constructs a pipeline based on provided
    arguments.

    The implemented policy does the following: for each input provided in
    `call`(), the policy first inputs a random number, if the number is < rate,
    the policy then selects a random layer from the provided list of `layers`.
    It then calls the `layer()` on the inputs. This is done
    `augmentations_per_image` times.

    This layer can be used to create custom policies resembling `RandAugment` or
    `AutoAugment`.

    Example:
    ```python
    # construct a list of layers
    layers = keras_cv.layers.RandAugment.get_standard_policy(
        value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
    )
    layers = layers[:4]  # slice out some layers you don't want for whatever
                           reason
    layers = layers + [keras_cv.layers.GridMask()]

    # create the pipeline.
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=layers, augmentations_per_image=3
    )

    augmented_images = pipeline(images)
    ```

    Args:
        layers: a list of `keras.Layers`. These are randomly inputs during
            augmentation to augment the inputs passed in `call()`. The layers
            passed should subclass `BaseImageAugmentationLayer`. Passing
            `layers=[]` would result in a no-op.
        augmentations_per_image: the number of layers to apply to each inputs in
            the `call()` method.
        rate: the rate at which to apply each augmentation. This is applied on a
            per augmentation bases, so if `augmentations_per_image=3` and
            `rate=0.5`, the odds an image will receive no augmentations is
            0.5^3, or 0.5*0.5*0.5.
        auto_vectorize: whether to use `tf.vectorized_map` or `tf.map_fn` to
            apply the augmentations. This offers a significant performance
            boost, but can only be used if all the layers provided to the
            `layers` argument support auto vectorization.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        layers,
        augmentations_per_image,
        rate=1.0,
        auto_vectorize=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs, seed=seed)
        self.augmentations_per_image = augmentations_per_image
        self.rate = rate
        self.layers = list(layers)
        self.auto_vectorize = auto_vectorize
        self.seed = seed

        self._random_choice = preprocessing.RandomChoice(
            layers=layers, auto_vectorize=auto_vectorize, seed=seed
        )

    def _augment(self, inputs):
        if self.layers == []:
            return inputs

        result = inputs
        for _ in range(self.augmentations_per_image):
            skip_augment = self._random_generator.uniform(
                shape=(), minval=0.0, maxval=1.0, dtype=tf.float32
            )
            result = tf.cond(
                skip_augment > self.rate,
                lambda: result,
                lambda: self._random_choice(result),
            )
        return result

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "augmentations_per_image": self.augmentations_per_image,
                "auto_vectorize": self.auto_vectorize,
                "rate": self.rate,
                "layers": self.layers,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        layers = config.pop("layers", None)
        if layers:
            if isinstance(layers[0], dict):
                layers = keras.utils.deserialize_keras_object(layers)
            config["layers"] = layers
        return cls(**config)
