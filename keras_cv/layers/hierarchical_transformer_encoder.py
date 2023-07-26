import math

from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.layers.efficient_multihead_attention import (
    EfficientMultiheadAttention,
)
from keras_cv.layers.regularization.drop_path import DropPath


@keras.saving.register_keras_serializable(package="keras_cv")
class HierarchicalTransformerEncoder(keras.layers.Layer):
    """
    Hierarchical transformer encoder block implementation as a Keras Layer.
    The layer uses `EfficientMultiheadAttention` as a `MultiHeadAttention` alternative for
    computational efficiency, and is meant to be used within the SegFormer architecture.

    References:
        - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021)
        - [Official PyTorch implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
        - [Ported from the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/layers/hierarchical_transformer_encoder.py)

    Args:
        project_dim: the dimensionality of the projection of the encoder, and
            output of the `EfficientMultiHeadAttention` layer. Due to the residual addition
            the input dimensionality has to be equal to the output dimensionality.
        num_heads: the number of heads for the `EfficientMultiHeadAttention` layer
        drop_prob: default 0.0, the probability of dropping a random sample using the `DropPath` layer.
        layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
            layers
        sr_ratio: default 1, the ratio to use within `EfficientMultiHeadAttention`. If set to > 1,
            a `Conv2D` layer is used to reduce the length of the sequence.

    Basic usage:

    ```
    project_dim = 1024
    num_heads = 4
    patch_size = 16

    encoded_patches = keras_cv.layers.OverlappingPatchingAndEmbedding(
    project_dim=project_dim, patch_size=patch_size)(img_batch)

    trans_encoded = keras_cv.layers.HierarchicalTransformerEncoder(project_dim=project_dim,
                                                                   num_heads=num_heads,
                                                                   sr_ratio=1)(encoded_patches)

    print(trans_encoded.shape) # (1, 3136, 1024)
    ```
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prop = drop_prob

        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = EfficientMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = DropPath(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = self.MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prop": self.drop_prop,
            }
        )
        return config

    class MixFFN(keras.layers.Layer):
        def __init__(self, channels, mid_channels):
            super().__init__()
            self.fc1 = keras.layers.Dense(mid_channels)
            self.dwconv = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=1,
                padding="same",
            )
            self.fc2 = keras.layers.Dense(channels)

        def call(self, x):
            x = self.fc1(x)
            shape = x.shape
            H, W = int(math.sqrt(shape[1])), int(math.sqrt(shape[1]))
            x = ops.reshape(x, (shape[0], H, W, shape[-1]))
            x = self.dwconv(x)
            x = ops.reshape(x, (shape[0], -1, shape[-1]))
            x = ops.nn.gelu(x)
            x = self.fc2(x)
            return x
