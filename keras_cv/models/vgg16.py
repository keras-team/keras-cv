# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""VGG16 model for KerasCV.
Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_cv.models import utils

class VGG16(keras.Model):

    @staticmethod
    def build_vgg_block(num_layers,
                    kernel_size,
                    stride,
                    activation,
                    padding,
                    max_pooling,
                    name,
                    ):
        block = keras.Sequential(name=name)
        for num in range(1, num_layers+1):
            block.add(layers.Conv2D(kernel_size,
                                    stride,
                                    activation=activation,
                                    padding=padding,
                                    name=f"{name}_conv{str(num)}"))
        if max_pooling:
            block.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name=f"{name}_pool"))

        return block

    def __init__(self,
                 include_rescaling,
                 include_top,
                 classes=None,
                 weights=None,
                 input_shape=(224, 224, 3),
                 pooling=None,
                 classifier_activation="softmax",
                 name="VGG16",
                 **kwargs,
                 ):

        self.input_shape = input_shape
        self.include_rescaling = include_rescaling
        self.include_top = include_top

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

        # inputs = utils.parse_model_inputs(input_shape, input_tensor)
        #
        # x = inputs
        if include_rescaling:
            self.rescaling_layer = layers.Rescaling(1 / 255.0)

        self.block_1 = build_vgg_block(num_layers=2,
                                  kernel_size=64,
                                  stride=(3,3),
                                  activation='relu',
                                  padding='same',
                                  max_pool=True,
                                  name='block1')

        self.block_2 = build_vgg_block(num_layers=2,
                                  kernel_size=128,
                                  stride=(3,3),
                                  activation='relu',
                                  padding='same',
                                  max_pool=True,
                                  name='block2')

        self.block_3 = build_vgg_block(num_layers=3,
                                  kernel_size=256,
                                  stride=(3,3),
                                  activation='relu',
                                  padding='same',
                                  max_pool=True,
                                  name='block3')

        self.block_4 = build_vgg_block(num_layers=3,
                                  kernel_size=512,
                                  stride=(3, 3),
                                  activation='relu',
                                  padding='same',
                                  max_pool=True,
                                  name='block4')

        self.block_5 = build_vgg_block(num_layers=3,
                                  kernel_size=512,
                                  stride=(3, 3),
                                  activation='relu',
                                  padding='same',
                                  max_pool=True,
                                  name='block5')

        # Block 1
        # x = layers.Conv2D(
        #     64, (3, 3), activation="relu", padding="same", name="block1_conv1"
        # )(x)
        # x = layers.Conv2D(
        #     64, (3, 3), activation="relu", padding="same", name="block1_conv2"
        # )(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
        #
        # # Block 2
        # x = layers.Conv2D(
        #     128, (3, 3), activation="relu", padding="same", name="block2_conv1"
        # )(x)
        # x = layers.Conv2D(
        #     128, (3, 3), activation="relu", padding="same", name="block2_conv2"
        # )(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
        #
        # # Block 3
        # x = layers.Conv2D(
        #     256, (3, 3), activation="relu", padding="same", name="block3_conv1"
        # )(x)
        # x = layers.Conv2D(
        #     256, (3, 3), activation="relu", padding="same", name="block3_conv2"
        # )(x)
        # x = layers.Conv2D(
        #     256, (3, 3), activation="relu", padding="same", name="block3_conv3"
        # )(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
        #
        # # Block 4
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block4_conv1"
        # )(x)
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block4_conv2"
        # )(x)
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block4_conv3"
        # )(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
        #
        # # Block 5
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block5_conv1"
        # )(x)
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block5_conv2"
        # )(x)
        # x = layers.Conv2D(
        #     512, (3, 3), activation="relu", padding="same", name="block5_conv3"
        # )(x)
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        if include_top:
            self.flatten_layer = layers.Flatten(name="flatten")(x)
            self.fc1_layer = layers.Dense(4096, activation="relu", name="fc1")(x)
            self.fc2_layer = layers.Dense(4096, activation="relu", name="fc2")(x)
            self.classification_layer = layers.Dense(
                classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                self.pool_layer = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":
                self.pool_layer = layers.GlobalMaxPooling2D()(x)

        # model = keras.Model(inputs, x, name=name, **kwargs)
        # if weights is not None:
        #     model.load_weights(weights)
        # return model

    def call(self, input_tensor):
        inputs = utils.parse_model_inputs(self.input_shape, input_tensor)
        x = inputs

        if self.include_rescaling is not None:
            x = self.rescaling_layer(x)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        if self.include_top:
            x = self.flatten_layer(x)
            x = self.fc1_layer(x)
            x = self.fc2_layer(x)
            out = self.classification_layer(x)
        else:
            out = self.pool_layer(x)

        return out