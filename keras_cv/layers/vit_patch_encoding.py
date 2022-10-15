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

from keras_cv.layers import ClassTokenizing


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class PatchEncoding(layers.Layer):
    """
    Layer to positionally embed and create a projection of patches made with `Patching` layer
    for Vision Transformers from:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

    Based on Khalid Salama's implementation for:
        - https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    args:
        - num_patches: the number of input patches to project
        - project_dim: the dimensionality of the project_dim
    Basic usage:

    ```
    patches = keras_cv.layers.Patching(patch_size)(batch_img)

    project_dim = 1024
    num_patches = patches.shape[1] # 196

    encoded_patches = keras_cv.layers.PatchEncoding(num_patches, project_dim)(patches)
    print(encoded_patches.shape) # (1, 196, 1024)
    ```
    """

    def __init__(self, num_patches, project_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.project_dim = layers.Dense(units=project_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=project_dim
        )

    def call(self, patch):
        # Add learnable class token before positional embedding
        patch = ClassTokenizing()(patch)
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
