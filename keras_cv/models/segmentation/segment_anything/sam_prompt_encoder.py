# Copyright 2023 The KerasCV Authors
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

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models.segmentation.segment_anything.sam_layers import (
    SAMLayerNormalization,
)


@keras.saving.register_keras_serializable(package="keras_cv")
class RandomFrequencyPositionalEmbeddings(keras.layers.Layer):
    def __init__(self, num_positional_features, scale, **kwargs):
        """Positional encoding using random spatial frequencies.

        This layer maps coordinates/points in 2D space to positional
        encodings using random spatial frequencies.

        Args:
            num_positional_features (int): Number of positional features
                in the output.
            scale (float): The standard deviation of the random frequencies.
        """
        super().__init__(**kwargs)
        self.num_positional_features = num_positional_features
        self.scale = scale
        init_func = lambda *a, **kw: self.scale * ops.random.normal(
            shape=(2, self.num_positional_features), dtype=self.dtype
        )
        self.positional_encoding_gaussian_matrix = self.add_weight(
            name="positional_emcoding_gaussian_matrix",
            shape=(2, self.num_positional_features),
            dtype=self.dtype,
            trainable=False,
            initializer=init_func,
        )

        self.built = True

    def __positional_encodings(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def call(self, size):
        """Generate a positional encoding for an image of any given size.

        Args:
            size (int): The size of the image.

        Returns:
            tensor: Positional encoding of the image.
        """
        H, W = size
        H, W = ops.cast(H, "int64"), ops.cast(W, "int64")
        grid = ops.ones(shape=(H, W), dtype=self.dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / ops.cast(H, self.dtype)
        x_embed = x_embed / ops.cast(W, self.dtype)
        return self.__positional_encodings(
            ops.stack([x_embed, y_embed], axis=-1)
        )

    def call_with_coords(self, coords_input, image_size):
        """Positionally encode points that are not normalized to `[0, 1]`.

        Args:
            coords_input (tensor): 2D coordinates/points to map.
            image_size (tuple[int, int]): Height and width of the image
                being prompted.

        Returns:
            tensor: Positional encodings of the normalized coordinates.
        """
        coords_normalized = ops.stack(
            [
                coords_input[..., 0] / image_size[1],
                coords_input[..., 1] / image_size[0],
            ],
            axis=-1,
        )
        return self.__positional_encodings(coords_normalized)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_positional_features": self.num_positional_features,
                "scale": self.scale,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_cv")
class PromptEncoder(keras.models.Model):
    """Prompt Encoder for the segment anything model.

    The prompt encoder generates encodings for three types of prompts:

    - Point prompts: Points on the image along with a label indicating whether
        the point is in the foreground (part of the mask) or in the background
        (not a part of the mask).
    - Box prompts: A batch of bounding boxes with format [(x1, y1), (x2, y2)]
        used to determine the location of the masks in the image.
    - Masks: An input mask can be passed to refine the positional embeddings
        for the output mask.

    First, the point prompts and box prompts are concatenated and positional
    encodings are generated using random spatial frequencies. A point is
    represented as the sum of a positional encoding of the point's location
    and one of two learned embeddings that indicate if the point is either in
    the foreground or background. A box is represented by an embedding pair:

    (1) the positional encoding of its top-left corner summed with a learned
    embedding representing "top-left corner" and
    (2) the same structure but using a learned embedding indicating
    "bottom-right corner".

    The box and point encodings are referred to as "sparse encodings"

    If a mask prompt is passed, a convolutional neural net is used to
    downscale it to generate "dense encodings". If no mask prompt is passed,
    an embedding layer is used instead to generate a "no mask" embedding.

    Args:
        embed_dim (int): The number of features in the output embeddings.
        image_embedding_size (int): The number of features in the image
            embeddings generated by an image encoder.
        input_image_size (tuple[int, int]): A tuple of the height and width
            of the image being prompted.
        mask_in_chans (int): The number of channels of the mask prompt.
        activation (str, optional): The activation to use in the mask
            downscaler neural net. Defaults to "gelu".

    References:
        - [Segment Anything](https://arxiv.org/abs/2304.02643)
    """

    def __init__(
        self,
        embed_dim,
        image_embedding_size,
        input_image_size,
        mask_in_chans,
        activation="gelu",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans
        self.activation = activation

        self.positional_embedding_layer = RandomFrequencyPositionalEmbeddings(
            num_positional_features=self.embed_dim // 2, scale=1
        )

        self.foreground_point_embed = keras.layers.Embedding(1, embed_dim)
        self.background_point_embed = keras.layers.Embedding(1, embed_dim)
        self.top_left_corner_embed = keras.layers.Embedding(1, embed_dim)
        self.bottom_right_corner_embed = keras.layers.Embedding(1, embed_dim)
        self.not_a_point_embed = keras.layers.Embedding(1, embed_dim)

        self.mask_downscaler = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    mask_in_chans // 4, kernel_size=2, strides=2
                ),
                SAMLayerNormalization(),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(mask_in_chans, kernel_size=2, strides=2),
                SAMLayerNormalization(),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(embed_dim, kernel_size=1),
            ]
        )
        self.mask_downscaler.build(
            [None, 4 * image_embedding_size[0], 4 * image_embedding_size[1], 1]
        )
        self.no_mask_embed = keras.layers.Embedding(1, embed_dim)

        for layer in [
            self.foreground_point_embed,
            self.background_point_embed,
            self.top_left_corner_embed,
            self.bottom_right_corner_embed,
            self.not_a_point_embed,
            self.no_mask_embed,
        ]:
            layer.build(None)

        self.built = True

    def get_dense_pe(self):
        """Get positional embeddings for the image being prompted.

        Returns:
            tensor: The positional embeddings of the image.
        """
        # convert the image_embedding_size to a tensor since keras core
        # expects the input type to be a symbolic/concrete tensor.
        return self.positional_embedding_layer(
            ops.convert_to_tensor(self.image_embedding_size, dtype="float32")
        )[None, ...]

    def __embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = ops.zeros((points.shape[0], 1, 2), dtype=self.dtype)
            padding_label = -ops.ones((labels.shape[0], 1), dtype=self.dtype)
            points = ops.concatenate([points, padding_point], axis=1)
            labels = ops.concatenate([labels, padding_label], axis=1)
        point_embeddings = self.positional_embedding_layer.call_with_coords(
            points, self.input_image_size
        )
        labels = ops.broadcast_to(labels[..., None], point_embeddings.shape)
        point_embeddings = ops.where(
            labels == 0,
            point_embeddings + self.background_point_embed.weights[0],
            point_embeddings + self.foreground_point_embed.weights[0],
        )
        point_embeddings = ops.where(
            labels == -1,
            # TODO: for whatever reason, ops.broadcast_to doesn't work here, so
            #       we instead use zeros_like to broadcast to the correct shape.
            self.not_a_point_embed.weights[0]
            + ops.zeros_like(point_embeddings),
            point_embeddings,
        )
        return point_embeddings

    def __embed_box(self, box):
        box = box + 0.5
        coords = ops.reshape(box, (-1, 2, 2))
        corner_embedding = self.positional_embedding_layer.call_with_coords(
            coords, self.input_image_size
        )
        top_left_embedding = (
            corner_embedding[:, 0, :] + self.top_left_corner_embed.weights[0]
        )
        bottom_right_embedding = (
            corner_embedding[:, 1, :]
            + self.bottom_right_corner_embed.weights[0]
        )
        corner_embedding = ops.stack(
            [top_left_embedding, bottom_right_embedding], axis=1
        )
        return corner_embedding

    def __embed_mask(self, mask):
        mask_embedding = self.mask_downscaler(mask)
        return mask_embedding

    def call(self, points=None, labels=None, box=None, mask=None):
        if points is not None:
            B = points.shape[0]
        elif box is not None:
            B = box.shape[0]
        elif mask is not None:
            B = mask.shape[0]
        else:
            raise ValueError("At least one of the inputs must not be None.")
        sparse_embeddings = ops.zeros((B, 0, self.embed_dim), dtype=self.dtype)
        if points is not None:
            if labels is None:
                raise ValueError("`labels` must also be provided with `points`")
            point_embeddings = self.__embed_points(
                points, labels, pad=(box is None)
            )
            sparse_embeddings = ops.concatenate(
                [sparse_embeddings, point_embeddings], axis=1
            )
        if box is not None:
            box_embeddings = self.__embed_box(box)
            sparse_embeddings = ops.concatenate(
                [sparse_embeddings, box_embeddings], axis=1
            )
        if mask is not None:
            dense_embeddings = self.__embed_mask(mask)
        else:
            dense_embeddings = ops.broadcast_to(
                ops.reshape(
                    self.no_mask_embed.weights[0], (1, 1, 1, self.embed_dim)
                ),
                shape=(
                    B,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                    self.embed_dim,
                ),
            )
        return sparse_embeddings, dense_embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "image_embedding_size": self.image_embedding_size,
                "input_image_size": self.input_image_size,
                "mask_in_chans": self.mask_in_chans,
                "activation": self.activation,
            }
        )
        return config
