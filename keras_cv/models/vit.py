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
  - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2) (ICLR 2021)
  - [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270) (CoRR 2021)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import TransformerEncoder
from keras_cv.layers.vit_layers import PatchEmbedding
from keras_cv.layers.vit_layers import Patching
from keras_cv.models import utils

MODEL_CONFIGS = {
    "ViT_Tiny_16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_S_16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_B_16": {
        "patch_size": 16,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_L_16": {
        "patch_size": 16,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_H_16": {
        "patch_size": 16,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_Tiny_32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 192,
        "mlp_dim": 768,
        "num_heads": 3,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_S_32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 384,
        "mlp_dim": 1536,
        "num_heads": 6,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_B_32": {
        "patch_size": 32,
        "transformer_layer_num": 12,
        "project_dim": 768,
        "mlp_dim": 3072,
        "num_heads": 12,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_L_32": {
        "patch_size": 32,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
    "ViT_H_32": {
        "patch_size": 32,
        "transformer_layer_num": 32,
        "project_dim": 1280,
        "mlp_dim": 5120,
        "num_heads": 16,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3,
    },
}

BASE_DOCSTRING = """Instantiates the {name} architecture.
    Reference:
        - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2) (ICLR 2021)
    This function returns a Keras {name} model.
    
    The naming convention of ViT models follows: ViT_Size_Patch-size (i.e. ViT_S_16).
    The following sizes were released in the original paper:
        - S (Small)
        - B (Base)
        - L (Large)
    But subsequent work from the same authors introduced:
        - Ti (Tiny)
        - H (Huge)
        
    The parameter configurations for all of these sizes, at patch sizes 16 and 32 are made available, following the naming convention
    laid out above.

    For transfer learning use cases, make sure to read the [guide to transfer
        learning & fine-tuning](https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of the
            network.  If provided, classes must be provided.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the output
                of the last convolutional block, and thus the output of the model will
                be a 2D tensor.
            - `max` means that global max pooling will be applied.
            - `token_pooling`, default, means that the token at the start of the 
                sequences is used instead of regular pooling.
        name: (Optional) name to pass to the model.  Defaults to "{name}".
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
    Returns:
      A `keras.Model` instance.
"""


def ViT(
    include_rescaling,
    include_top,
    name="ViT",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
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
    """Instantiates the ViT architecture.

    Args:
        include_rescaling: whether or not to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
            name: string, model name.
        include_top: whether to include the fully-connected
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
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True.
                    mlp_dim:
        project_dim: the latent dimensionality to be projected into in the output
            of each stacked transformer encoder
        activation: the activation function to use in the first `layers.Dense` layer
            in the MLP head of the transformer encoder
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
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        **kwargs: Pass-through keyword arguments to `tf.keras.Model`.

    Returns:
      A `keras.Model` instance.
    """

    if weights and not tf.io.gfile.exists(weights):
        raise ValueError(
            "The `weights` argument should be either `None` or the path to the "
            "weights file to be loaded. Weights file not found at location: {weights}"
        )

    if include_top and not classes:
        raise ValueError(
            "If `include_top` is True, you should specify `classes`. "
            f"Received: classes={classes}"
        )

    if include_top and pooling:
        raise ValueError(
            f"`pooling` must be `None` when `include_top=True`."
            f"Received pooling={pooling} and include_top={include_top}. "
        )

    inputs = utils.parse_model_inputs(input_shape, input_tensor)
    x = inputs

    if include_rescaling:
        x = layers.Rescaling(1 / 255.0)(x)

    patches = Patching(patch_size)(x)
    encoded_patches = PatchEmbedding(project_dim)(patches)

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
        output = layers.Lambda(lambda rep: rep[:, 0])(output)
        output = layers.Dense(classes, activation=classifier_activation)(output)

    elif pooling == "token_pooling":
        output = layers.Lambda(lambda rep: rep[:, 0])(output)
    elif pooling == "avg":
        output = layers.GlobalAveragePooling1D()(output)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def ViT_Tiny_16(
    include_rescaling,
    include_top,
    name="ViT_Tiny_16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_Tiny_16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_Tiny_16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_Tiny_16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_Tiny_16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_Tiny_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_Tiny_16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_Tiny_16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_Tiny_16"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_S_16(
    include_rescaling,
    include_top,
    name="ViT_S_16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_S_16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_S_16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_B_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_S_16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_S_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_S_16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_S_16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_S_16"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_B_16(
    include_rescaling,
    include_top,
    name="ViT_B_16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_B_16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_B_16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_B_16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_B_16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_B_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_B_16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_B_16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_B_16"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_L_16(
    include_rescaling,
    include_top,
    name="ViT_L_16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_L_16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_L_16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_L_16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_L_16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_L_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_L_16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_L_16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_L_16"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_H_16(
    include_rescaling,
    include_top,
    name="ViT_H_16",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_H_16 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_H_16"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_H_16"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_H_16"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_H_16"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_H_16"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_H_16"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_H_16"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_Tiny_32(
    include_rescaling,
    include_top,
    name="ViT_Tiny_32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_Tiny_32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_Tiny_32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_Tiny_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_Tiny_32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_Tiny_32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_Tiny_32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_Tiny_32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_Tiny_32"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_S_32(
    include_rescaling,
    include_top,
    name="ViT_S_32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_S_32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_S_32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_S_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_S_32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_S_32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_S_32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_S_32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_S_32"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_B_32(
    include_rescaling,
    include_top,
    name="ViT_B_32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_B_32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_B_32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_B_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_B_32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_B_32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_B_32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_B_32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_B_32"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_L_32(
    include_rescaling,
    include_top,
    name="ViT_L_32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_L_32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_L_32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_L_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_L_32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_L_32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_L_32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_L_32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_L_32"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


def ViT_H_32(
    include_rescaling,
    include_top,
    name="ViT_H_32",
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):
    """Instantiates the ViT_H_32 architecture."""

    return ViT(
        include_rescaling,
        include_top,
        name=name,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT_H_32"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT_H_32"]["transformer_layer_num"],
        project_dim=MODEL_CONFIGS["ViT_H_32"]["project_dim"],
        mlp_dim=MODEL_CONFIGS["ViT_H_32"]["mlp_dim"],
        num_heads=MODEL_CONFIGS["ViT_H_32"]["num_heads"],
        mlp_dropout=MODEL_CONFIGS["ViT_H_32"]["mlp_dropout"],
        attention_dropout=MODEL_CONFIGS["ViT_H_32"]["attention_dropout"],
        classifier_activation=classifier_activation,
        **kwargs,
    )


setattr(ViT_Tiny_16, "__doc__", BASE_DOCSTRING.format(name="ViT_Tiny_16"))
setattr(ViT_S_16, "__doc__", BASE_DOCSTRING.format(name="ViT_S_16"))
setattr(ViT_B_16, "__doc__", BASE_DOCSTRING.format(name="ViT_B_16"))
setattr(ViT_L_16, "__doc__", BASE_DOCSTRING.format(name="ViT_L_16"))
setattr(ViT_H_16, "__doc__", BASE_DOCSTRING.format(name="ViT_H_16"))
setattr(ViT_Tiny_32, "__doc__", BASE_DOCSTRING.format(name="ViT_Tiny_32"))
setattr(ViT_S_32, "__doc__", BASE_DOCSTRING.format(name="ViT_S_32"))
setattr(ViT_B_32, "__doc__", BASE_DOCSTRING.format(name="ViT_B_32"))
setattr(ViT_L_32, "__doc__", BASE_DOCSTRING.format(name="ViT_L_32"))
setattr(ViT_H_32, "__doc__", BASE_DOCSTRING.format(name="ViT_H_32"))
