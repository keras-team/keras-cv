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
"""ViT (Vision Transformer) models for Keras.
Reference:
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
    (ICLR 2021)
  - [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
    (CoRR 2021)
"""  # noqa: E501

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.src.layers import TransformerEncoder
from keras_cv.src.layers.vit_layers import PatchingAndEmbedding
from keras_cv.src.models.legacy import utils
from keras_cv.src.models.legacy.weights import parse_weights

MODEL_CONFIGS = {
    "ViTTiny16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL16": {
        "patch_size": 16,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH16": {
        "patch_size": 16,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTTiny32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTS32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTB32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.0,
        "attention_dropout": 0.0,
    },
    "ViTL32": {
        "patch_size": 32,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
    "ViTH32": {
        "patch_size": 32,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.1,
        "attention_dropout": 0.0,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
        (ICLR 2021)
    This function returns a Keras {name} model.

    The naming convention of ViT models follows: ViTSize_Patch-size
        (i.e. ViTS16).
    The following sizes were released in the original paper:
        - S (Small)
        - B (Base)
        - L (Large)
    But subsequent work from the same authors introduced:
        - Ti (Tiny)
        - H (Huge)

    The parameter configurations for all of these sizes, at patch sizes 16 and
    32 are made available, following the naming convention laid out above.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(scale=1./255.0)`
            layer. Note that ViTs expect an input range of `[0..1]` if rescaling
            isn't used. Regardless of whether you supply `[0..1]` or the input
            is rescaled to `[0..1]`, the inputs will further be rescaled to
            `[-1..1]`.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, num_classes must be provided.
        num_classes: optional int, number of classes to classify images into,
            only to be specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights
            (e.g. 'imagenet/classification') (see available pre-trained weights
            in weights.py). Note that the 'imagenet' weights only work on an
            input shape of (224, 224, 3) due to the input shape dependent
            patching and flattening logic.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
            - `token_pooling`, default, means that the token at the start of the
                sequences is used instead of regular pooling.
        name: (Optional) name to pass to the model, defaults to "{name}".
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
    Returns:
      A `keras.Model` instance.
"""  # noqa: E501


@keras.utils.register_keras_serializable(package="keras_cv.models")
class ViT(keras.Model):
    """Instantiates the ViT architecture.

    Args:
        mlp_dim: the dimensionality of the hidden Dense layer in the transformer
            MLP head
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        name: string, model name.
        include_top: bool, whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            or the path to the weights file to be loaded.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
            - `token_pooling`, default, means that the token at the start of the
                sequences is used instead of regular pooling.
        num_classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
                    mlp_dim:
        project_dim: the latent dimensionality to be projected into in the
            output of each stacked transformer encoder
        activation: the activation function to use in the first `layers.Dense`
            layer in the MLP head of the transformer encoder
        attention_dropout: the dropout rate to apply to the `MultiHeadAttention`
            in each transformer encoder
        mlp_dropout: the dropout rate to apply between `layers.Dense` layers
            in the MLP head of the transformer encoder
        num_heads: the number of heads to use in the `MultiHeadAttention` layer
            of each transformer encoder
        transformer_layer_num: the number of transformer encoder layers to stack
            in the Vision Transformer
        patch_size: the patch size to be supplied to the Patching layer to turn
            input images into a flattened sequence of patches
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.
        **kwargs: Pass-through keyword arguments to `keras.Model`.
    """

    def __init__(
        self,
        include_rescaling,
        include_top,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        num_classes=None,
        patch_size=None,
        transformer_layer_num=None,
        num_heads=None,
        mlp_dropout=None,
        attention_dropout=None,
        activation=None,
        project_dim=None,
        mlp_dim=None,
        classifier_activation="softmax",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path "
                "to the weights file to be loaded. Weights file not found at "
                "location: {weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. "
                f"Received: num_classes={num_classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

        # The previous layer rescales [0..255] to [0..1] if applicable
        # This one rescales [0..1] to [-1..1] since ViTs expect [-1..1]
        x = layers.Rescaling(scale=1.0 / 0.5, offset=-1.0, name="rescaling_2")(
            x
        )

        encoded_patches = PatchingAndEmbedding(project_dim, patch_size)(x)
        encoded_patches = layers.Dropout(mlp_dropout)(encoded_patches)

        for _ in range(transformer_layer_num):
            encoded_patches = TransformerEncoder(
                project_dim=project_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                mlp_dropout=mlp_dropout,
                attention_dropout=attention_dropout,
                activation=activation,
            )(encoded_patches)

        output = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        if include_top:
            output = output[:, 0]
            output = layers.Dense(
                num_classes, activation=classifier_activation
            )(output)

        elif pooling == "token_pooling":
            output = output[:, 0]
        elif pooling == "avg":
            output = layers.GlobalAveragePooling1D()(output)

        # Create model.
        super().__init__(inputs=inputs, outputs=output, **kwargs)

        if weights is not None:
            self.load_weights(weights)

        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.transformer_layer_num = transformer_layer_num
        self.num_heads = num_heads
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.project_dim = project_dim
        self.mlp_dim = mlp_dim
        self.classifier_activation = classifier_activation

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "name": self.name,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "num_classes": self.num_classes,
            "patch_size": self.patch_size,
            "transformer_layer_num": self.transformer_layer_num,
            "num_heads": self.num_heads,
            "mlp_dropout": self.mlp_dropout,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
            "project_dim": self.project_dim,
            "mlp_dim": self.mlp_dim,
            "classifier_activation": self.classifier_activation,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def ViTTiny16(
    *,
    include_rescaling,
    include_top,
    name="ViTTiny16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTTiny16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vittiny16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTTiny16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny16"][
            "transformer_layer_num"
        ],
        project_dim=MODEL_CONFIGS["ViTTiny16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTS16(
    *,
    include_rescaling,
    include_top,
    name="ViTS16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTS16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vits16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTS16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTB16(
    *,
    include_rescaling,
    include_top,
    name="ViTB16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTB16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitb16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTB16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTL16(
    *,
    include_rescaling,
    include_top,
    name="ViTL16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTL16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitl16"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTL16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTH16(
    *,
    include_rescaling,
    include_top,
    name="ViTH16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTH16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTH16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH16"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTTiny32(
    *,
    include_rescaling,
    include_top,
    name="ViTTiny32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTTiny32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTTiny32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTTiny32"][
            "transformer_layer_num"
        ],
        project_dim=MODEL_CONFIGS["ViTTiny32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTTiny32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTTiny32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTTiny32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTTiny32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTS32(
    *,
    include_rescaling,
    include_top,
    name="ViTS32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTS32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vits32"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTS32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTS32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTS32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTS32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTS32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTS32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTS32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTB32(
    *,
    include_rescaling,
    include_top,
    name="ViTB32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTB32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=parse_weights(weights, include_top, "vitb32"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTB32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTB32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTB32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTB32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTB32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTB32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTB32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTL32(
    *,
    include_rescaling,
    include_top,
    name="ViTL32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTL32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTL32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTL32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTL32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTL32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTL32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTL32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTL32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViTH32(
    *,
    include_rescaling,
    include_top,
    name="ViTH32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    num_classes=None,
    activation=keras.activations.gelu,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViTH32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        num_classes=num_classes,
        patch_size=MODEL_CONFIGS["ViTH32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViTH32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViTH32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViTH32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViTH32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViTH32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViTH32"]["attention_dropout"],
        activation=activation,
        classifier_activation=classifier_activation,
        **kwargs,
    )


setattr(ViTTiny16, "__doc__", BASE_DOCSTRING.format(name="ViTTiny16"))
setattr(ViTS16, "__doc__", BASE_DOCSTRING.format(name="ViTS16"))
setattr(ViTB16, "__doc__", BASE_DOCSTRING.format(name="ViTB16"))
setattr(ViTL16, "__doc__", BASE_DOCSTRING.format(name="ViTL16"))
setattr(ViTH16, "__doc__", BASE_DOCSTRING.format(name="ViTH16"))
setattr(ViTTiny32, "__doc__", BASE_DOCSTRING.format(name="ViTTiny32"))
setattr(ViTS32, "__doc__", BASE_DOCSTRING.format(name="ViTS32"))
setattr(ViTB32, "__doc__", BASE_DOCSTRING.format(name="ViTB32"))
setattr(ViTL32, "__doc__", BASE_DOCSTRING.format(name="ViTL32"))
setattr(ViTH32, "__doc__", BASE_DOCSTRING.format(name="ViTH32"))
