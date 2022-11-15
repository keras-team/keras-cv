import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow import keras

from keras_cv.layers.vit_layers import Patching
from keras_cv.layers.vit_layers import PatchEmbedding
from keras_cv.layers import TransformerEncoder

from keras_cv.models import utils

MODEL_CONFIGS = {
    "ViT_B_32": {
        "patch_size":  32,
        "transformer_layer_num" : 12,
        "project_dim" : 768,
        "mlp_dim": 3072,
        "num_heads" : 12,
        "mlp_dropout" : 0.3,
        "attention_dropout" : 0.3
    },
    "ViT_L_32": {
        "patch_size": 32,
        "transformer_layer_num": 24,
        "project_dim": 1024,
        "mlp_dim": 4096,
        "num_heads": 16,
        "mlp_dropout": 0.3,
        "attention_dropout": 0.3
    },
}

def ViT(
    include_rescaling,
    include_top,
    name="ViT_B_32",
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
        encoded_patches = TransformerEncoder(project_dim=project_dim,
                                             mlp_dim=mlp_dim,
                                             num_heads=num_heads,
                                             mlp_dropout=mlp_dropout,
                                             attention_dropout=attention_dropout,
                                             activation=activation)(encoded_patches)

    output = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

    if include_top:
        output = layers.Lambda(lambda rep: rep[:, 0])(output)
        output = layers.Dense(classes, activation=classifier_activation)(output)

    else:
      if pooling=='token_pooling':
        output = layers.Lambda(lambda rep: rep[:, 0])(output)
      else:
        output = layers.GlobalAveragePooling1D()(output)

    model = keras.Model(inputs=inputs, outputs=output)
    return model


def ViT_B_32(
    include_rescaling,
    include_top,
    name="ViT_B_32",
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
    project_dim=None,
    mlp_dim=None,
    classifier_activation="softmax",
    **kwargs,
):

    return ViT(
        include_rescaling,
        include_top,
        name="ViT_B_32",
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

