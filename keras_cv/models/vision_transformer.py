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
"""ResNet models for KerasCV.
Reference:
  - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2015)
  - [Based on the original keras.applications ResNet](https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py)
"""

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow import keras

from keras_cv.transformers import mlp_ffn
from keras_cv.layers import Patching
from keras_cv.layers import PatchEncoder
from keras_cv.layers import TransformerEncoder

from keras_cv.models import utils

MODEL_CONFIGS = {
    "ViT": {
        "patch_size":  2,
        "transformer_layer_num" : 2,
        #"num_patches" : [],
        "project_dim" : 64,
        "head_units" : [2048, 1024],
        "num_heads" : 4,
        "dropout" : 0.1,
        "activation" : tf.nn.gelu()
    },
}

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
    num_patches=None,
    num_heads=None,
    dropout=None,
    activation=None,
    head_units=None,
    project_dim=None,
    classes=None,
    classifier_activation="softmax",
    **kwargs,
):

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
    # Encode patches.
    """Temp: calc num_patches"""
    num_patches = (inputs.shape[0] // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, project_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layer_num):
        x = TransformerEncoder(project_dim=project_dim,
                               num_heads=num_heads,
                               dropout=dropout,
                               activation=activation,
                               transformer_units=None)(encoded_patches)

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(x)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features =  mlp_ffn(representation, hidden_units=head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(classes)(features)

    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def ViT1(
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
    num_patches=None,
    num_heads=None,
    dropout=None,
    activation=None,
    project_dim=None,
    classifier_activation="softmax",
    **kwargs,
):

    return ViT(
        include_rescaling,
        include_top,
        name="ViT",
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classes=classes,
        patch_size=MODEL_CONFIGS["ViT"]["patch_size"],
        transformer_layer_num=MODEL_CONFIGS["ViT"]["transformer_layer_num"],
        num_patches=MODEL_CONFIGS["ViT"]["num_patches"],
        project_dim=MODEL_CONFIGS["ViT"]["project_dim"],
        num_heads=MODEL_CONFIGS["ViT"]["num_heads"],
        dropout=MODEL_CONFIGS["ViT"]["dropout"],
        activation=MODEL_CONFIGS["ViT"]["activation"],
        classes=classes,
        classifier_activation="softmax",
        **kwargs,
    )
