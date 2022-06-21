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

"""AdaIN model for KerasCV.

Reference:
  - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
  - [Based on the keras code examples: Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain/)
"""

import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers


class NeuralStyleTransferWithAdaIN(tf.keras.Model):
    """A neural style transfer model using adaptive instance normalization (AdaIN).
    Reference:
	  - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
	  - [Based on the keras code examples: Neural Style Transfer with AdaIN](https://keras.io/examples/generative/adain/)
    """

    def __init__(self, image_size=(None, None), epsilon=1e-5 ,**kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.epsilon = epsilon
        self.encoder = self._get_encoder()
        self.decoder = self._get_decoder()
        
    def call(self, inputs):
        style, content = inputs
        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded)

        # Generate the neural style transferred image.
        return self.decoder(t)
        
    def _get_encoder(self):
    	vgg19 = keras.applications.VGG19(
		include_top=False,
		weights="imagenet",
		input_shape=(*self.image_size, 3),
    	)
    	vgg19.trainable = False
    	mini_vgg19 = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    	inputs = layers.Input([*self.image_size, 3])
    	mini_vgg19_out = mini_vgg19(inputs)
    	return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")
    	
    def _get_decoder(self):
        config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
        decoder = keras.Sequential(
            [
                layers.InputLayer((*self.image_size, 512)),
                layers.Conv2D(filters=512, **config),
                layers.UpSampling2D(),
                layers.Conv2D(filters=256, **config),
                layers.Conv2D(filters=256, **config),
                layers.Conv2D(filters=256, **config),
                layers.Conv2D(filters=256, **config),
                layers.UpSampling2D(),
                layers.Conv2D(filters=128, **config),
                layers.Conv2D(filters=128, **config),
                layers.UpSampling2D(),
                layers.Conv2D(filters=64, **config),
                layers.Conv2D(
                    filters=3,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    activation="sigmoid",
                ),
            ]
        )
        return decoder	
    
    def _get_mean_std(self, x):
        """Computes the mean and standard deviation of a tensor."""
	axes = [1, 2]
	mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
	standard_deviation = tf.sqrt(variance + self.epsilon)
	return mean, standard_deviation

    def _ada_in(self, style, content):
        """Computes the AdaIn feature map.

        Args:
            style: The style feature map.
            content: The content feature map.

        Returns:
            The AdaIN feature map.
        """
        content_mean, content_std = self._get_mean_std(content)
        style_mean, style_std = self.get_mean_std(style)
        t = style_std * (content - content_mean) / content_std + style_mean
        return t

