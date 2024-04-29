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
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@keras_cv_export("keras_cv.layers.RandomChoice")
class RandomChoice(BaseImageAugmentationLayer):
    """RandomChoice constructs a pipeline based on provided arguments.

    The implemented policy does the following: for each input provided in
    `call`(), the policy selects a random layer from the provided list of
    `layers`. It then calls the `layer()` on the inputs.

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
    pipeline = keras_cv.layers.RandomChoice(layers=layers)

    augmented_images = pipeline(images)
    ```

    Args:
        layers: a list of `keras.Layers`. These are randomly inputs during
            augmentation to augment the inputs passed in `call()`. The layers
            passed should subclass `BaseImageAugmentationLayer`.
        auto_vectorize: whether to use `tf.vectorized_map` or `tf.map_fn` to
            apply the augmentations. This offers a significant performance
            boost, but can only be used if all the layers provided to the
            `layers` argument support auto vectorization.
        batchwise: Boolean, whether to pass entire batches to the
            underlying layer. When set to `True`, each batch is passed to a
            single layer, instead of each sample to an independent layer. This
            is useful when using `MixUp()`, `CutMix()`, `Mosaic()`, etc.
            Defaults to `False`.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self,
        layers,
        auto_vectorize=False,
        batchwise=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs, seed=seed)
        self.layers = layers
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed

    def _curry_call_layer(self, inputs, layer):
        def call_layer():
            return layer(inputs)

        return call_layer

    def _batch_augment(self, inputs):
        if self.batchwise:
            return self._augment(inputs)
        else:
            return super()._batch_augment(inputs)

    def _augment(self, inputs, *args, **kwargs):
        selected_op = self._random_generator.uniform(
            (), minval=0, maxval=len(self.layers), dtype=tf.int32
        )
        # Warning:
        # Do not replace the currying function with a lambda.
        # Originally we used a lambda, but due to Python's
        # lack of loop level scope this causes unexpected
        # behavior running outside of graph mode.
        #
        # Autograph has an edge case where the behavior of Python for loop
        # variables is inconsistent between Python and graph execution.
        # By using a list comprehension and currying, we mitigate
        # our code against both of these cases.
        branch_fns = [
            (i, self._curry_call_layer(inputs, layer))
            for (i, layer) in enumerate(self.layers)
        ]
        return tf.switch_case(
            branch_index=selected_op,
            branch_fns=branch_fns,
            default=lambda: inputs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "auto_vectorize": self.auto_vectorize,
                "seed": self.seed,
                "batchwise": self.batchwise,
            }
        )
        return config
