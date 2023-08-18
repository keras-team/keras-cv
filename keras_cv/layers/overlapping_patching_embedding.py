from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops


@keras_cv_export("keras_cv.layers.OverlappingPatchingAndEmbedding")
class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, project_dim=32, patch_size=7, stride=4, **kwargs):
        """
        Overlapping Patching and Embedding layer. Differs from `PatchingAndEmbedding` in that the patch size
        does not affect the sequence length. It's fully derived from the `stride` parameter.
        Additionally, no positional embedding is done as part of the layer - only a projection using a `Conv2D` layer.

        References:
            - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021)
            - [Official PyTorch implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
            - [Ported from the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/layers/hierarchical_transformer_encoder.py)

        Args:
            project_dim: the dimensionality of the projection of the encoder, and
                output of the `MultiHeadAttention`
            num_heads: the number of heads for the `MultiHeadAttention` layer
            drop_prob: default 0.0, the probability of dropping a random sample using the `DropPath` layer.
            layer_norm_epsilon: default 1e-06, the epsilon for `LayerNormalization`
                layers
            sr_ratio: default 1, the ratio to use within `SegFormerMultiheadAttention`. If set to > 1,
                a `Conv2D` layer is used to reduce the length of the sequence.

        Basic usage:

        ```
        project_dim = 1024
        patch_size = 16

        encoded_patches = keras_cv.layers.OverlappingPatchingAndEmbedding(
        project_dim=project_dim, patch_size=patch_size)(img_batch)

        print(encoded_patches.shape) # (1, 3136, 1024)
        ```
        """
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.patch_size = patch_size
        self.stride = stride

        self.proj = keras.layers.Conv2D(
            filters=project_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="same",
        )
        self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        x = self.proj(x)
        # B, H, W, C
        shape = x.shape
        x = ops.reshape(x, (-1, shape[1] * shape[2], shape[3]))
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "patch_size": self.patch_size,
                "stride": self.stride,
            }
        )
        return config
