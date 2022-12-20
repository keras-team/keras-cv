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

from keras_cv.models.vgg16 import VGG16
from keras_cv.models.weights import parse_weights


def get_vgg_layers(block_name, input_shape):
    vgg16 = VGG16(include_rescaling=False, include_top='False', input_shape=(224,224,3))
    return vgg16.get_layer(f"{block_name}").output, vgg16

def fcn8(n_classes):
    vgg_layer_pool_3, vgg = get_vgg_layers("block3_pool", (224,224,3))
    vgg_layer_pool_4, vgg = get_vgg_layers("block4_pool", (224,224,3))
    vgg_layer_pool_5, vgg = get_vgg_layers("block5_pool", (224,224,3))

    x1 = vgg_layer_pool_5
    x1 = tf.keras.layers.Conv2D(4096, (7,7), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu',padding='same')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal')(x1)
    x1 = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(x1)

    x2 = vgg_layer_pool_4
    x2 = tf.keras.layers.Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal')(x2)

    x1 = tf.keras.layers.Add()([x1, x2])
    x1 = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(x1)

    x2 = vgg_layer_pool_3
    x2 = tf.keras.layers.Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal')(x2)

    x1 = tf.keras.layers.Add()([x2, x1])
    x1 = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(16, 16),  strides=(8, 8), use_bias=False)(x1)

    model = tf.keras.Model(inputs = vgg.inputs, outputs = x1)
    return model

def fcn16(n_classes):
    vgg_layer_pool_4, vgg = get_vgg_layers("block4_pool", (224,224,3))
    vgg_layer_pool_5, vgg = get_vgg_layers("block5_pool", (224,224,3))

    x1 = vgg_layer_pool_5
    x1 = tf.keras.layers.Conv2D(4096, (7,7), activation='relu', padding='same')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(4096, (1, 1), activation='relu',padding='same')(x1)
    x1 = tf.keras.layers.Dropout(0.5)(x1)
    x1 = tf.keras.layers.Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal')(x1)
    x1 = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(4, 4),  strides=(2, 2), use_bias=False)(x1)

    x2 = vgg_layer_pool_4
    x2 = tf.keras.layers.Conv2D(n_classes,  (1, 1), kernel_initializer='he_normal')(x2)

    x1 = tf.keras.layers.Add()([x1, x2])
    x1 = tf.keras.layers.Conv2DTranspose(n_classes, kernel_size=(32, 32),  strides=(16, 16), use_bias=False)(x1)
    model = tf.keras.Model(inputs = vgg.inputs, outputs = x1)
    return model

def fcn32(num_classes):
    vgg_layer_pool_5, vgg = get_vgg_layers("block5_pool", (224,224,3))
    x = tf.keras.layers.Conv2D(4096, (7,7), activation='relu', padding='same')(vgg_layer_pool_5)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(4096, (1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(num_classes, (1,1), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(num_classes, kernel_size=(64, 64),  strides=(32, 32), use_bias=False)(x)
    model = tf.keras.Model(inputs = vgg.inputs, outputs = x)
    return model
    