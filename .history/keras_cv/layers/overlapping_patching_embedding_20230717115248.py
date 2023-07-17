# Copyright 2023 David Landup
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
from torch import nn


class __OverlappingPatchingAndEmbeddingPT(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=32,
        patch_size=7,
        stride=4,
        name=None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class __OverlappingPatchingAndEmbeddingTF(tf.keras.layers.Layer):
    def __init__(
        self, in_channels=3, out_channels=32, patch_size=7, stride=4, **kwargs
    ):
        super().__init__(**kwargs)
        self.proj = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=patch_size,
            strides=stride,
            padding="same",
        )
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.proj(x)
        # B, H, W, C
        shape = tf.shape(x)
        x = tf.reshape(x, [-1, shape[1] * shape[2], shape[3]])
        x = self.norm(x)
        return x, shape[1], shape[2]


LAYER_BACKBONES = {
    "tensorflow": __OverlappingPatchingAndEmbeddingTF,
    "pytorch": __OverlappingPatchingAndEmbeddingPT,
}


def OverlappingPatchingAndEmbedding(
    in_channels=3,
    out_channels=32,
    patch_size=7,
    stride=4,
    backend=None,
    name=None,
):
    """
    ViT-inspired PatchingAndEmbedding, modified to merge overlapping patches for the SegFormer architecture.

    Reference:
        - ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/pdf/2105.15203v2.pdf)


    Args:
        in_channels: the number of channels in the input tensor
        out_channels: the projection dimensionality
        patch_size: the patch size/kernel size to apply in the convolutional layer used to patchify
        stride: the stride to apply in the convolutional layer used to patchify
        backend: the backend framework to use

    Basic usage:

    ```
    inp = torch.rand(1, 3, 224, 224)
    output, H, W = deepvision.layers.OverlappingPatchingAndEmbedding(in_channels=3,
                                                                     out_channels=64,
                                                                     patch_size=7,
                                                                     stride=4,
                                                                     backend='pytorch')(inp)
    print(output.shape) # torch.Size([1, 3136, 64])


    inp = tf.random.uniform(1, 224, 224, 3)
    output, H, W = deepvision.layers.OverlappingPatchingAndEmbedding(in_channels=3,
                                                                     out_channels=64,
                                                                     patch_size=7,
                                                                     stride=4,
                                                                     backend='tensorflow')(inp)
    print(output.shape) # (1, 3136, 64)
    ```
    """

    layer_class = LAYER_BACKBONES.get(backend)
    if layer_class is None:
        raise ValueError(
            f"Backend not supported: {backend}. Supported backbones are {LAYER_BACKBONES.keys()}"
        )

    layer = layer_class(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        stride=stride,
        name=name,
    )

    return layer
