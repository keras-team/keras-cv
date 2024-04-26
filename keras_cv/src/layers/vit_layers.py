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

from keras_cv.src.api_export import keras_cv_export


@keras_cv_export("keras_cv.layers.PatchingAndEmbedding")
class PatchingAndEmbedding(layers.Layer):
    """
    Layer to patchify images, prepend a class token, positionally embed and
    create a projection of patches for Vision Transformers

    The layer expects a batch of input images and returns batches of patches,
    flattened as a sequence and projected onto `project_dims`. If the height and
    width of the images aren't divisible by the patch size, the supplied padding
    type is used (or 'VALID' by default).

    Reference:
        An Image is Worth 16x16 Words: Transformers for Image Recognition at
        Scale by Alexey Dosovitskiy et al. (https://arxiv.org/abs/2010.11929)

    Args:
        project_dim: the dimensionality of the project_dim
        patch_size: the patch size
        padding: default 'VALID', the padding to apply for patchifying images

    Returns:
        Patchified and linearly projected input images, including a prepended
        learnable class token with shape (batch, num_patches+1, project_dim)

    Example:

    ```
    images = #... batch of images
    encoded_patches = keras_cv.layers.PatchingAndEmbedding(
        project_dim=project_dim,
        patch_size=patch_size)(patches)
    print(encoded_patches.shape) # (1, 197, 1024)
    ```
    """

    def __init__(self, project_dim, patch_size, padding="VALID", **kwargs):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.patch_size = patch_size
        self.padding = padding
        if patch_size < 0:
            raise ValueError(
                "The patch_size cannot be a negative number. Received "
                f"{patch_size}"
            )
        if padding not in ["VALID", "SAME"]:
            raise ValueError(
                f"Padding must be either 'SAME' or 'VALID', but {padding} was "
                "passed."
            )
        self.projection = layers.Conv2D(
            filters=self.project_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding=self.padding,
        )

    def build(self, input_shape):
        self.class_token = self.add_weight(
            shape=[1, 1, self.project_dim], name="class_token", trainable=True
        )
        self.num_patches = (
            input_shape[1]
            // self.patch_size
            * input_shape[2]
            // self.patch_size
        )
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1, output_dim=self.project_dim
        )

    def call(
        self,
        images,
        interpolate=False,
        interpolate_width=None,
        interpolate_height=None,
        patch_size=None,
    ):
        """Calls the PatchingAndEmbedding layer on a batch of images.
        Args:
            images: A `tf.Tensor` of shape [batch, width, height, depth]
            interpolate: A `bool` to enable or disable interpolation
            interpolate_height: An `int` representing interpolated height
            interpolate_width: An `int` representing interpolated width
            patch_size: An `int` representing the new patch size if
                interpolation is used

        Returns:
            `A tf.Tensor` of shape [batch, patch_num+1, embedding_dim]
        """
        # Turn images into patches and project them onto `project_dim`
        patches = self.projection(images)
        patch_shapes = tf.shape(patches)
        patches_flattened = tf.reshape(
            patches,
            shape=(
                patch_shapes[0],
                patch_shapes[-2] * patch_shapes[-2],
                patch_shapes[-1],
            ),
        )

        # Add learnable class token before linear projection and positional
        # embedding
        flattened_shapes = tf.shape(patches_flattened)
        class_token_broadcast = tf.cast(
            tf.broadcast_to(
                self.class_token,
                [flattened_shapes[0], 1, flattened_shapes[-1]],
            ),
            dtype=patches_flattened.dtype,
        )
        patches_flattened = tf.concat(
            [class_token_broadcast, patches_flattened], 1
        )
        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)

        if interpolate and None not in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            (
                interpolated_embeddings,
                class_token,
            ) = self.__interpolate_positional_embeddings(
                self.position_embedding(positions),
                interpolate_width,
                interpolate_height,
                patch_size,
            )
            addition = patches_flattened + interpolated_embeddings
            encoded = tf.concat([class_token, addition], 1)
        elif interpolate and None in (
            interpolate_width,
            interpolate_height,
            patch_size,
        ):
            raise ValueError(
                "`None of `interpolate_width`, `interpolate_height` and "
                "`patch_size` cannot be None if `interpolate` is True"
            )
        else:
            encoded = patches_flattened + self.position_embedding(positions)
        return encoded

    def __interpolate_positional_embeddings(
        self, embedding, height, width, patch_size
    ):
        """
        Allows for pre-trained position embedding interpolation. This trick
        allows you to fine-tune a ViT on higher resolution images than it was
        trained on.

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

        reshaped_embeddings = tf.reshape(
            tensor=interpolated_embeddings, shape=(1, -1, dimensionality)
        )

        # linear_projection = self.linear_projection(reshaped_embeddings)
        # addition = linear_projection + reshaped_embeddings

        # return tf.concat([class_token, addition], 1)
        return reshaped_embeddings, class_token

    def get_config(self):
        config = {
            "project_dim": self.project_dim,
            "patch_size": self.patch_size,
            "padding": self.padding,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
