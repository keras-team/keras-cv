from keras_cv.backend import keras


@keras.saving.register_keras_serializable(package="keras_cv")
class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, out_channels=32, patch_size=7, stride=4, **kwargs):
        super().__init__(**kwargs)
        self.proj = keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=patch_size,
            strides=stride,
            padding="same",
        )
        self.norm = keras.layers.LayerNormalization()

    def call(self, x):
        x = self.proj(x)
        # B, H, W, C
        shape = x.shape
        x = keras.ops.reshape(x, (-1, shape[1] * shape[2], shape[3]))
        x = self.norm(x)
        return x
