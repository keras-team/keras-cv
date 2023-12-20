from deepvision.layers.clip_patching_and_embedding import (
    CLIPPatchingAndEmbedding,
)
from deepvision.layers.residual_transformer_encoder import (
    ResidualTransformerEncoder,
)
from deepvision.utils.utils import parse_model_inputs

from keras_cv.backend import keras
from keras_cv.backend import ops


class __CLIPImageEncoder(keras.Model):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_tensor=None,
        **kwargs,
    ):
        inputs = keras.layers.Input(
            tensor=input_tensor, shape=(input_resolution, input_resolution, 3)
        )
        x = inputs

        x = CLIPPatchingAndEmbedding(
            width=width,
            patch_size=patch_size,
            input_resolution=input_resolution,
            backend="tensorflow",
        )(x)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = ops.transpose(x, perm=(1, 0, 2))
        x = ResidualTransformerEncoder(
            width, layers, heads, backend="tensorflow"
        )(x)
        x = ops.transpose(x, perm=(1, 0, 2))

        x = keras.layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])

        proj = keras.layers.Dense(output_dim)
        x = proj(x)

        output = x

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )
