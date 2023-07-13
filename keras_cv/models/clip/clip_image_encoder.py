# encode images
from collections import OrderedDict
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from tensorflow.keras import layers
from torch import nn

from deepvision.layers.clip_patching_and_embedding import CLIPPatchingAndEmbedding
from deepvision.layers.residual_transformer_encoder import ResidualTransformerEncoder
from deepvision.utils.utils import parse_model_inputs


class __CLIPImageEncoderTF(tf.keras.Model):
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
        inputs = tf.keras.layers.Input(
            tensor=input_tensor, shape=(input_resolution, input_resolution, 3)
        )
        x = inputs

        x = CLIPPatchingAndEmbedding(
            width=width,
            patch_size=patch_size,
            input_resolution=input_resolution,
            backend="tensorflow",
        )(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

        x = tf.transpose(x, perm=(1, 0, 2))
        x = ResidualTransformerEncoder(width, layers, heads, backend="tensorflow")(x)
        x = tf.transpose(x, perm=(1, 0, 2))

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])

        proj = tf.keras.layers.Dense(output_dim)
        x = proj(x)

        output = x

        super().__init__(
            inputs=inputs,
            outputs=output,
            **kwargs,
        )