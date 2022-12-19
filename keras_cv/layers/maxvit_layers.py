"""
- GridPartitioning layer (me)
- WindowPartitioning layer (me)
- MaxViTTransformerEncoder layer (me?)
- RelativeMultiHeadAttention layer (you)
- SqueezeExcite layer (potentially don't need it because it's already part of MBConv)
- MBConv layer (already have it)
- MaxViT layer (MBConv + Block-Attention + FFN + Grid-Attention + FFN)
"""

import tensorflow as tf
from tf.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class GridPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L842
    Partition the input feature maps into non-overlapping windows.
        Args:
          features: [B, H, W, C] feature maps.
        Returns:
          Partitioned features: [B, nH, nW, wSize, wSize, c].
        Raises:
          ValueError: If the feature map sizes are not divisible by window sizes.
    """

    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size

    def call(self, input):
        _, h, w, c = input.shape
        grid_size = self.grid_size
        if h % grid_size != 0 or w % grid_size != 0:
            raise ValueError(
                f"Feature map sizes {(h, w)} "
                f"not divisible by window size ({grid_size})."
            )
        features = tf.reshape(
            input, (-1, grid_size, h // grid_size, grid_size, w // grid_size, c)
        )
        features = tf.transpose(features, (0, 2, 4, 1, 3, 5))
        features = tf.reshape(features, (-1, grid_size, grid_size, c))
        return features

    def get_config(self):
        config = {"grid_size": self.grid_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
