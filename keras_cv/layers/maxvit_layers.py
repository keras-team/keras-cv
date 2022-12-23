"""
- MaxViTTransformerEncoder layer
- RelativeMultiHeadAttention layer
- SqueezeExcite layer (potentially don't need it because it's already part of MBConv)
- MBConv layer (already have it)
- MaxViT layer (MBConv + Block-Attention + FFN + Grid-Attention + FFN)
"""

import tensorflow as tf
from tensorflow.keras import layers

from keras_cv.layers.mbconv import MBConvBlock


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class WindowPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L805
    Partition the input feature maps into non-overlapping windows.

        Args:
          inputs: [B, H, W, C] feature maps.
        Returns:
          A `tf.Tensor`: Partitioned features: [B, nH, nW, wSize, wSize, c].
        Raises:
          ValueError: If the feature map sizes are not divisible by window sizes.


    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    layer = keras_cv.layers.WindowPartitioning(window_size=64)
    outputs = layer(inputs)
    outputs.shape # TensorShape([16, 64, 64, 3])
    ```
    """

    def __init__(self, window_size, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def call(self, input):
        _, h, w, c = input.shape
        window_size = self.window_size

        if h % window_size != 0 or w % window_size != 0:
            raise ValueError(
                f"Feature map sizes {(h, w)} "
                f"not divisible by window size ({window_size})."
            )

        features = tf.reshape(
            input, (-1, h // window_size, window_size, w // window_size, window_size, c)
        )
        features = tf.transpose(features, (0, 1, 3, 2, 4, 5))
        features = tf.reshape(features, (-1, window_size, window_size, c))
        return features

    def get_config(self):
        config = {"window_size": self.window_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UnWindowPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L832
    Reverses the operation of the WindowPartitioning layer.

        Args:
          window_size: the window size used while performing WindowPartitioning
          height: the desired height to stitch the feature map to
          width: the desired width to stitch the feature map to
        Returns:
          A `tf.Tensor`: Stiched feature map: [B, H, W, C].

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    window_layer = keras_cv.layers.WindowPartitioning(window_size=64)
    outputs = window_layer(inputs)

    unwindow_layer = keras_cv.layers.UnWindowPartitioning(window_size=64, height=256, width=256)
    unwindow_outputs = unwindow_layer(outputs)
    unwindow_outputs.shape # TensorShape([1, 256, 256, 3])

    import numpy as np
    np.testing.assert_array_equal(unwindow_outputs, inputs)
    ```
    """

    def __init__(self, window_size, height, width, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.height = height
        self.width = width

    def call(self, input):
        features = tf.reshape(
            input,
            [
                -1,
                self.height // self.window_size,
                self.width // self.window_size,
                self.window_size,
                self.window_size,
                input.shape[-1],
            ],
        )
        return tf.reshape(
            tf.transpose(features, (0, 1, 3, 2, 4, 5)),
            [-1, self.height, self.width, features.shape[-1]],
        )

    def get_config(self):
        config = {
            "window_size": self.window_size,
            "height": self.height,
            "width": self.width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


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

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    layer = keras_cv.layers.GridPartitioning(grid_size=64)
    outputs = layer(inputs)
    outputs.shape # TensorShape([16, 64, 64, 3])
    ```
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


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class UnGridPartitioning(layers.Layer):
    """
    Based on: https://github.com/google-research/maxvit/blob/2e06a7f1f70c76e64cd3dabe5cd1b8c1a23c9fb7/maxvit/models/maxvit.py#L867
    Reverses the operation of the GridPartitioning layer.

        Args:
          grid_size: the grid size used while performing GridPartitioning
          height: the desired height to stitch the feature map to
          width: the desired width to stitch the feature map to
        Returns:
          A `tf.Tensor`: Stiched feature map: [B, H, W, C].

    Basic usage:

    ```
    inputs = tf.random.normal([1, 256, 256, 3])

    grid_layer = keras_cv.layers.GridPartitioning(grid_size=64)
    outputs = grid_layer(inputs)

    ungrid_layer = keras_cv.layers.UnGridPartitioning(grid_size=64, height=256, width=256)
    ungrid_outputs = ungrid_layer(outputs)
    ungrid_outputs.shape # TensorShape([1, 256, 256, 3])

    import numpy as np
    np.testing.assert_array_equal(ungrid_outputs, inputs)
    ```
    """

    def __init__(self, grid_size, height, width, **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.height = height
        self.width = width

    def call(self, input):
        features = tf.reshape(
            input,
            [
                -1,
                self.height // self.grid_size,
                self.width // self.grid_size,
                self.grid_size,
                self.grid_size,
                input.shape[-1],
            ],
        )
        return tf.reshape(
            tf.transpose(features, (0, 3, 1, 4, 2, 5)),
            [-1, self.height, self.width, features.shape[-1]],
        )

    def get_config(self):
        config = {
            "grid_size": self.grid_size,
            "height": self.height,
            "width": self.width,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTStem(layers.Layer):
    # Conv blocks
    def __init__(
        self,
        filters: list = [64,64],
        kernel_size: tuple = (3,3),
        kernel_initializer = tf.random_normal_initializer(stddev=0.02),
        bias_initializer = tf.zeros_initializer,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert len(filters) == 2

        self.conv1 = layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=2,
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='stem_conv_0'
        )

        self.batch_norm = layers.BatchNormalization()
        self.gelu = layers.Activation('gelu')

        self.conv2 = layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name='stem_conv_1'
        )

    def call(self, input):
        # Creates a stem for the MaxViT model.
        x = self.conv1(input)
        x = self.batch_norm(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return x

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTTransformerEncoder(layers.Layer):
    # Attention + FFN (LN + Attention + Residual + LN + MLP)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        # ...
        return input

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class RelativeMultiHeadAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        # ...
        return input

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class MaxViTBlock(layers.Layer):
    # (MBConv + Block-Attention (Block-SA+FFN) + Grid-Attention (Grid-SA+FFN))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        # ...
        return input

    def get_config(self):
        # config = {"...": self....}
        # base_config = super().get_config()
        # return dict(list(base_config.items()) + list(config.items()))
        return super().get_config()
