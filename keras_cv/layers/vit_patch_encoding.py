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
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class PatchEncoding(layers.Layer):
    """
    Layer to positionally embed and create a projection of patches made with `Patching` layer
    for Vision Transformers from:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

    args:
        - num_patches: the number of input patches to project
        - project_dim: the dimensionality of the project_dim
    """
    def __init__(self, num_patches, project_dim):
        super().__init__()
        self.num_patches = num_patches
        self.project_dim = layers.Dense(units=project_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=project_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.project_dim(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = {
            "num_patches": self.num_patches,
            "project_dim": self.project_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))