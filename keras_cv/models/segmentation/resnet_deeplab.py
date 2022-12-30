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

import tensorflow as tf
from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from tensorflow import keras
from tensorflow.keras import layers

import keras_cv

IMAGE_SIZE = 512

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DeeplabV3(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras_cv.models.ResNet50V2(
        include_rescaling=True, include_top=False, weights="imagenet",
        input_tensor=model_input
    )
    x = resnet50.get_layer("v2_stack_3_block3_out").output
    x = SpatialPyramidPooling(dilation_rates=[6, 12, 18], dropout=0.5)(x)

    input_a = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="nearest",
    )(x)
    x = input_a

    x = layers.Dropout(.2)(x)
    model_output = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1), 
        padding="same", 
        kernel_regularizer=keras.regularizers.l2(0.0001),
        activation="softmax")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))

    resnet50 = keras_cv.models.ResNet50V2(
        include_rescaling=True, include_top=False, weights="imagenet",
        input_tensor=model_input
    )
    x = resnet50.get_layer("v2_stack_3_block3_out").output
    x = SpatialPyramidPooling(dilation_rates=[6, 12, 18], dropout=0.5)(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("v2_stack_0_block3_out").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    x = layers.Dropout(.2)(x)
    model_output = layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1), 
        padding="same", 
        kernel_regularizer=keras.regularizers.l2(0.0001),
        activation="softmax")(x)
    return keras.Model(inputs=model_input, outputs=model_output)

