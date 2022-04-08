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
    def __init__(
        self,
        layers,
        augmentations_per_image,
        rate=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs, seed=seed)
        self.augmentations_per_image = augmentations_per_image
        self.rate = rate
        self.layers = layers
        self.auto_vectorize = False
        self.seed = seed

    def _augment(self, sample):
        result = sample.copy()
        for _ in range(self.augmentations_per_image):
            result = self._single_augmentation(result)
        return result

    def _single_augmentation(self, sample):
        skip_augment = self._random_generator.random_uniform(
            shape=(), minval=0.0, maxval=1.0, dtype=tf.float32
        )
        if skip_augment > self.rate:
            return sample

        selected_op = self._random_generator.random_uniform(
            (), minval=0, maxval=len(self.layers), dtype=tf.int32
        )

        branch_fns = []
        for (i, layer) in enumerate(self.layers):
            branch_fns.append((i, lambda: layer(sample)))

        return tf.switch_case(
            branch_index=selected_op,
            branch_fns=branch_fns,
            default=lambda: sample,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "augmentations_per_image": self.augmentations_per_image,
                "rate": self.rate,
                "layers": self.seed,
            }
        )
        return config
