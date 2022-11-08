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

import math

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class Patching(layers.Layer):
    """
    Layer to turn images into a sequence of patches for Vision Transformers from:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)
    Based on Khalid Salama's implementation for:
        - https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    The layer expects a batch of input images and returns batches of patches.

    args:
        - patch_size: the size (patch_size, patch_size) of each patch created from the image
    returns:
        - batch of patchified input images.

    Basic usage:

    ```
    img = url_to_array(url)
    batch_img = np.expand_dims(img, 0)

    patch_size = 16
    patches = keras_cv.layers.Patching(patch_size)(batch_img)
    print(f"Image size: {batch_img.shape}")                # Image size: (1, 224, 224, 3)
    print(f"Patch size: {(patch_size, patch_size)}") # Patch size: (16, 16)
    print(f"Patches: {patches.shape[1]}")            # Patches: 196
    ```
    """

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(
            patches, [batch_size, patches.shape[-2] * patches.shape[-2], patch_dims]
        )
        return patches

    def get_config(self):
        config = {
            "patch_size": self.patch_size,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class PatchEmbedding(layers.Layer):
    """
    Layer to concat a class token, positionally embed and create a projection of patches made with the `Patching` layer
    for Vision Transformers from:
        - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
        by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

    Based on Khalid Salama's implementation for:
        - https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py

    args:
        - project_dim: the dimensionality of the project_dim

    Basic usage:

    ```
    patches = keras_cv.layers.Patching(patch_size)(batch_img)

    project_dim = 1024
    num_patches = patches.shape[1] # 196

    encoded_patches = keras_cv.layers.PatchEmbedding(project_dim=project_dim)(patches)
    print(encoded_patches.shape) # (1, 197, 1024)
    ```
    """

    def __init__(self, project_dim, **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.linear_projection = layers.Dense(self.project_dim)

    def build(self, input_shape):
        self.class_token = tf.random.normal([1, 1, input_shape[-1]], name="class_token")
        self.num_patches = input_shape[1]
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1, output_dim=self.project_dim
        )

    def call(
        self,
        patch,
        interpolate=False,
        interpolate_width=None,
        interpolate_height=None,
        patch_size=None,
    ):
        # Add learnable class token before linear projection and positional embedding
        patch_shape = tf.shape(patch)
        class_token_broadcast = tf.cast(
            tf.broadcast_to(
                self.class_token,
                [patch_shape[0], 1, patch_shape[-1]],
            ),
            dtype=patch.dtype,
        )
        patch = tf.concat([class_token_broadcast, patch], 1)
        # num_patches + class token
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)

        if interpolate and None not in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            encoded = self.linear_projection(
                patch
            ) + self.interpolate_positional_embeddings(
                self.position_embedding(positions),
                interpolate_width,
                interpolate_height,
                patch_size,
            )
        elif interpolate and None in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            raise ValueError(
                "`None of `interpolate_width`, `interpolate_height` and `patch_size` cannot be None if `interpolate` is True"
            )
        else:
            encoded = self.linear_projection(patch) + self.position_embedding(positions)
        return encoded

    def interpolate_positional_embeddings(self, embedding, height, width, patch_size):
        """
        Allows for pre-trained position embedding interpolation. This trick allows you to fine-tune a ViT
        on higher resolution images than it was trained on.
        Based on:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_tf_vit.py
        """

        dimensionality = embedding.shape[-1]

        class_token = tf.expand_dims(embedding[:1, :], 0)
        patch_positional_embeddings = embedding[1:, :]

        h0 = height // patch_size
        w0 = width // patch_size

        new_shape = tf.constant(int(math.sqrt(self.num_patches)))

        interpolated_embeddings = tf.image.resize(
            images=tf.reshape(
                patch_positional_embeddings,
                shape=(
                    1,
                    new_shape,
                    new_shape,
                    dimensionality,
                ),
            ),
            size=(h0, w0),
            method="bicubic",
        )

        interpolated_embeddings = tf.reshape(
            tensor=interpolated_embeddings, shape=(1, -1, dimensionality)
        )
        return tf.concat([class_token, interpolated_embeddings], 1)

    def get_config(self):
        config = {
            "num_patches": self.num_patches,
            "project_dim": self.project_dim,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
