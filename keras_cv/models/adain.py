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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_mini_vgg19_encoder(include_rescaling, image_size):
    inputs = layers.Input([*image_size, 3])
    x = inputs
    if include_rescaling:
         x = keras.applications.vgg19.preprocess_input(x)
    backbone = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    x = backbone(x)
    backbone.trainable = False
    return keras.Model(inputs, backbone.get_layer("block4_conv1").output, name="mini_vgg19")


def get_adain_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    inputs = keras.Input((None, None, 512))
    x = layers.Conv2D(filters=512, **config)(inputs)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(filters=256, **config)(x)
    x = layers.Conv2D(filters=256, **config)(x)
    x = layers.Conv2D(filters=256, **config)(x)
    x = layers.Conv2D(filters=256, **config)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(filters=128, **config)(x)
    x = layers.Conv2D(filters=128, **config)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(filters=64, **config)(x)
    x = layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid")(x)
    outputs = tf.clip_by_value(x * 255., 0., 255.0)            
    return keras.Model(inputs, outputs, name="decoder")    


def get_loss_net(include_rescaling, image_size=(None, None)):
    inputs = layers.Input([*image_size, 3])
    x = inputs
    if include_rescaling:
    	x = keras.applications.vgg19.preprocess_input(x) 
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_tensor=x
    )
    x = vgg19(x)
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(x, outputs)
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")


def get_mean_std(x):
    """Computes the mean and standard deviation of a tensor."""
    axes = [1, 2]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + 1e-5)
    return mean, standard_deviation

@tf.keras.utils.register_keras_serializable(package="keras_cv")
class AdaIN(layers.Layer):
    def __init__(self, name='AdaIN', **kwargs):
        super(AdaIN, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        """Computes the AdaIn feature map.

        Args:
            style: The style feature map.
            content: The content feature map.

        Returns:
            The AdaIN feature map.
        """
        style, content = inputs
        content_mean, content_std = get_mean_std(content)
        style_mean, style_std = get_mean_std(style)
        t = style_std * (content - content_mean) / content_std + style_mean
        return t


def create_adain_model(include_rescaling, image_size):
    encoder = get_mini_vgg19_encoder(include_rescaling, image_size)
    decoder = get_adain_decoder()
    adain_layer = AdaIn()
    style_input = layers.Input([*image_size, 3])
    style_encoded = encoder(style_input)
    content_input = layers.Input([*image_size, 3])
    content_encoded = encoder(content_input)

    adain_feature_map = adain_layer((style_encoded, content_encoded))
    content_decoded = decoder(adain_feature_map)
        
    model=keras.Model([style_input, content_input], content_decoded, name="adain_style_transfer")
    return encoder, decoder, model
    


class AdaInTrainer(keras.Model):
    def __init__(self, include_rescaling, image_size=(None, None), style_weight=4.0, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.encoder, self.decoder, self.model = create_adain_model(include_rescaling, image_size)
        self.loss_net = get_loss_net(include_rescaling, image_size)
        self.style_weight = style_weight
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        
    def call(self, inputs):
        style, content = inputs
        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = self.adain_layer((style_encoded, content_encoded))

        # Generate the neural style transferred image.
        return self.decoder(t), t
        

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            loss_style, loss_content, total_loss = self.__compute_loss(inputs)

        # Compute gradients and optimize the decoder.
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def test_step(self, inputs):
        loss_style, loss_content, total_loss = self.__compute_loss(inputs)
        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }
        
    def __compute_loss(self, inputs):
    	style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0
	reconstructed_image, t = self(inputs)
	
	# Compute the losses.
	reconstructed_vgg_features = self.loss_net(reconstructed_image)
	style_vgg_features = self.loss_net(style)
	loss_content = self.compiled_loss(t, reconstructed_vgg_features[-1])
	for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
	    mean_inp, std_inp = get_mean_std(inp)
	    mean_out, std_out = get_mean_std(out)
	    loss_style += self.compiled_loss(mean_inp, mean_out) + self.compiled_loss(std_inp, std_out)

	loss_style = self.style_weight * loss_style
	total_loss = loss_content + loss_style
	return loss_content, loss_style, total_loss

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]    

