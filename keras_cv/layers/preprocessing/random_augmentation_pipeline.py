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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RandomAugmentationPipeline(
    tf.keras.__internal__.layers.BaseImageAugmentationLayer
):
    """RandomAugmentationPipeline constructs a pipeline based on provided arguments.

    The implemented policy does the following: for each inputs provided in `call`(), the
    policy first inputss a random number, if the number is < rate, the policy then
    selects a random layer from the provided list of `layers`.  It then calls the
    `layer()` on the inputs.  This is done `augmentations_per_image` times.

    This layer can be used to create custom policies resembling `RandAugment` or
    `AutoAugment`.

    Usage:
    ```python
    # construct a list of layers
    layers = keras_cv.layers.RandAugment.get_standard_policy(
        value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3
    )
    layers = layers[:4]  # slice out some layers you don't want for whatever reason
    layers = layers + [keras_cv.layers.GridMask()]

    # create the pipeline.
    pipeline = keras_cv.layers.RandomAugmentationPipeline(
        layers=layers, augmentations_per_image=3
    )

    augmented_images = pipeline(images)
    ```

    Args:
        layers: a list of `keras.Layers`.  These are randomly inputs during
            augmentation to augment the inputss passed in `call()`.  The layers passed
            should subclass `BaseImageAugmentationLayer`.
        augmentations_per_image: the number of layers to apply to each inputs in the
            `call()` method.
        rate: the rate at which to apply each augmentation.  This is applied on a per
            augmentation bases, so if `augmentations_per_image=3` and `rate=0.5`, the
            odds an image will receive no augmentations is 0.5^3, or 0.5*0.5*0.5.
        auto_vectorize: whether or not to use `tf.vectorized_map` or `tf.map_fn` to
            apply the augmentations.  This offers a significant performance boost, but
            can only be used if all of the layers provided to the `layers` argument
            support auto vectorization.
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
        super().__init__(**kwargs, seed=seed, force_generator=True)
        self.augmentations_per_image = augmentations_per_image
        self.rate = rate
        self.layers = layers
        self.auto_vectorize = auto_vectorize
        self.seed = seed

    def _augment(self, inputs):
        result = inputs
        for _ in range(self.augmentations_per_image):
            result = self._single_augmentation(result)
        return result

    def _curry_call_layer(self, inputs, layer):
        def call_layer():
            return layer(inputs)

        return call_layer

    def _single_augmentation(self, inputs):
        def _augment():
            selected_op = self._random_generator.random_uniform(
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

        skip_augment = self._random_generator.random_uniform(
            shape=(), minval=0.0, maxval=1.0, dtype=tf.float32
        )
        return tf.cond(skip_augment > self.rate, lambda: inputs, _augment)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "augmentations_per_image": self.augmentations_per_image,
                "rate": self.rate,
                "layers": self.layers,
                "seed": self.seed,
            }
        )
        return config
