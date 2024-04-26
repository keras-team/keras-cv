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
from keras_cv.src.layers.preprocessing.base_image_augmentation_layer import (
    BaseImageAugmentationLayer,
)


@keras_cv_export("keras_cv.layers.RandomApply")
class RandomApply(BaseImageAugmentationLayer):
    """Apply provided layer to random elements in a batch.

    Args:
        layer: a keras `Layer` or `BaseImageAugmentationLayer`. This layer will
            be applied to randomly chosen samples in a batch. Layer should not
            modify the size of provided inputs.
        rate: controls the frequency of applying the layer. 1.0 means all
            elements in a batch will be modified. 0.0 means no elements will be
            modified. Defaults to 0.5.
        batchwise: (Optional) bool, whether to pass entire batches to the
            underlying layer. When set to true, only a single random sample is
            drawn to determine if the batch should be passed to the underlying
            layer. This is useful when using `MixUp()`, `CutMix()`, `Mosaic()`,
            etc.
        auto_vectorize: bool, whether to use tf.vectorized_map or tf.map_fn for
            batched input. Setting this to True might give better performance
            but currently doesn't work with XLA. Defaults to False.
        seed: integer, controls random behaviour.

    Example:
    ```
    # Let's declare an example layer that will set all image pixels to zero.
    zero_out = keras.layers.Lambda(lambda x: {"images": 0 * x["images"]})

    # Create a small batch of random, single-channel, 2x2 images:
    images = tf.random.stateless_uniform(shape=(5, 2, 2, 1), seed=[0, 1])
    print(images[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.08216608, 0.40928006],
    #         [0.39318466, 0.3162533 ]],
    #
    #        [[0.34717774, 0.73199546],
    #         [0.56369007, 0.9769211 ]],
    #
    #        [[0.55243933, 0.13101244],
    #         [0.2941643 , 0.5130266 ]],
    #
    #        [[0.38977218, 0.80855536],
    #         [0.6040567 , 0.10502195]],
    #
    #        [[0.51828027, 0.12730157],
    #         [0.288486  , 0.252975  ]]], dtype=float32)>

    # Apply the layer with 50% probability:
    random_apply = RandomApply(layer=zero_out, rate=0.5, seed=1234)
    outputs = random_apply(images)
    print(outputs[..., 0])
    # <tf.Tensor: shape=(5, 2, 2), dtype=float32, numpy=
    # array([[[0.        , 0.        ],
    #         [0.        , 0.        ]],
    #
    #        [[0.34717774, 0.73199546],
    #         [0.56369007, 0.9769211 ]],
    #
    #        [[0.55243933, 0.13101244],
    #         [0.2941643 , 0.5130266 ]],
    #
    #        [[0.38977218, 0.80855536],
    #         [0.6040567 , 0.10502195]],
    #
    #        [[0.        , 0.        ],
    #         [0.        , 0.        ]]], dtype=float32)>

    # We can observe that the layer has been randomly applied to 2 out of 5
    samples.
    ```
    """

    def __init__(
        self,
        layer,
        rate=0.5,
        batchwise=False,
        auto_vectorize=False,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)

        if not (0 <= rate <= 1.0):
            raise ValueError(
                f"rate must be in range [0, 1]. Received rate: {rate}"
            )

        self._layer = layer
        self._rate = rate
        self.auto_vectorize = auto_vectorize
        self.batchwise = batchwise
        self.seed = seed
        self.built = True

    def _should_augment(self):
        return self._random_generator.uniform(shape=()) > 1.0 - self._rate

    def _batch_augment(self, inputs):
        if self.batchwise:
            # batchwise augmentations
            if self._should_augment():
                return self._layer(inputs)
            else:
                return inputs
        # non-batchwise augmentations
        return super()._batch_augment(inputs)

    def _augment(self, inputs):
        if self._should_augment():
            return self._layer(inputs)
        else:
            return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rate": self._rate,
                "layer": self._layer,
                "seed": self.seed,
                "batchwise": self.batchwise,
                "auto_vectorize": self.auto_vectorize,
            }
        )
        return config
