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

from keras_cv import core
from keras_cv.layers import preprocessing as cv_preprocessing
from keras_cv.utils import preprocessing as preprocessing_utils


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class AugmentationPipelines(tf.keras.__internal__.layers.BaseImageAugmentationLayer):
    """
    """

    def __init__(
        self,
        layers,
        distortions,
        seed=None,
        rate=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs, seed=seed)
        self.distortions = distortions
        self.magnitude = float(magnitude)
        self.seed = seed
        self.rate = rate

        self.layers = layers
        self.auto_vectorize = False

    def _augment(self, sample):
        augmented_sample = sample
        for _ in range(self.distortions):
            selected_op = self._random_generator.random_uniform(
                (), minval=0, maxval=len(self.layers) + 1, dtype=tf.int32
            )
            branch_fns = []
            for (i, layer) in enumerate(self.layers):
                branch_fns.append((i, lambda: layer(augmented_sample)))

            sample_augmented_by_this_layer = tf.switch_case(
                branch_index=selected_op,
                branch_fns=branch_fns,
                default=lambda: augmented_sample,
            )
            if self.rate is not None:
                augmented_sample = tf.cond(
                    self._random_generator.random_uniform(shape=(), dtype=tf.float32)
                    < self.rate,
                    lambda: sample_augmented_by_this_layer,
                    lambda: augmented_sample,
                )
            augmented_sample = sample_augmented_by_this_layer
        return augmented_sample

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
                "distortions": self.distortions,
                "magnitude": self.magnitude,
                "rate": self.rate
            }
        )
        return config
