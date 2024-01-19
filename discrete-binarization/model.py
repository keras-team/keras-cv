import math

import keras
from keras import layers
from keras import ops

import keras_cv


def get_backbone_model(input_shape=(640, 640, 3)):
    resnet = keras_cv.models.ResNet50Backbone.from_preset(
        "resnet50_imagenet", input_shape=input_shape, load_weights=False
    )
    levels = ["P2", "P3", "P4", "P5"]
    layer_names = [resnet.pyramid_level_inputs[level] for level in levels]
    items = zip(levels, layer_names)
    outputs = {key: resnet.get_layer(name).output for key, name in items}
    backbone = keras.Model(resnet.inputs, outputs=outputs)
    return backbone


def FPNModel(out_channels, **kwargs):
    def apply(inputs):
        # c2, c3, c4, c5 = inputs
        c2 = inputs["P2"]
        c3 = inputs["P3"]
        c4 = inputs["P4"]
        c5 = inputs["P5"]
        in2 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(c2)
        in3 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(c3)
        in4 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(c4)
        in5 = layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(c5)
        out4 = layers.Add()([layers.UpSampling2D()(in5), in4])
        out3 = layers.Add()([layers.UpSampling2D()(out4), in3])
        out2 = layers.Add()([layers.UpSampling2D()(out3), in2])
        p5 = layers.Conv2D(
            out_channels // 4, kernel_size=3, padding="same", use_bias=False
        )(in5)
        p4 = layers.Conv2D(
            out_channels // 4, kernel_size=3, padding="same", use_bias=False
        )(out4)
        p3 = layers.Conv2D(
            out_channels // 4, kernel_size=3, padding="same", use_bias=False
        )(out3)
        p2 = layers.Conv2D(
            out_channels // 4, kernel_size=3, padding="same", use_bias=False
        )(out2)
        p5 = layers.UpSampling2D((8, 8))(p5)
        p4 = layers.UpSampling2D((4, 4))(p4)
        p3 = layers.UpSampling2D((2, 2))(p3)

        fused = layers.Concatenate(axis=-1)([p5, p4, p3, p2])
        return fused

    return apply


def Head(in_channels, kernel_list=[3, 2, 2], **kwargs):
    def apply(inputs):
        x = layers.Conv2D(
            in_channels // 4,
            kernel_size=kernel_list[0],
            padding="same",
            use_bias=False,
        )(inputs)
        x = layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(1e-4),
            gamma_initializer=keras.initializers.Constant(1.0),
        )(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(
            in_channels // 4,
            kernel_size=kernel_list[1],
            strides=2,
            padding="valid",
            bias_initializer=keras.initializers.RandomUniform(
                minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
                maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
            ),
        )(x)
        x = layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(1e-4),
            gamma_initializer=keras.initializers.Constant(1.0),
        )(x)
        x = layers.ReLU()(x)
        x = layers.Conv2DTranspose(
            1,
            kernel_size=kernel_list[2],
            strides=2,
            padding="valid",
            activation="sigmoid",
            bias_initializer=keras.initializers.RandomUniform(
                minval=-1.0 / math.sqrt(in_channels // 4 * 1.0),
                maxval=1.0 / math.sqrt(in_channels // 4 * 1.0),
            ),
        )(x)
        return x

    return apply


def step_function(x, y, k=50):
    return 1.0 / 1 + ops.exp(-k * (x - y))


def DBHead(in_channels, k=50, **kwargs):
    def apply(inputs, training):
        probability_maps = Head(in_channels, **kwargs)(inputs)
        if not training:
            return probability_maps

        threshold_maps = Head(in_channels, **kwargs)(inputs)
        binary_maps = step_function(probability_maps, threshold_maps)
        y = layers.Concatenate(axis=-1)(
            [probability_maps, threshold_maps, binary_maps]
        )
        return y

    return apply
