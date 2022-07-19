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


def get_mini_vgg19_encoder(image_size):
    vgg19 = keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    vgg19.trainable = False
    return keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output, name="mini_vgg19")


def get_adain_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),
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


def get_loss_net(image_size=(None, None)):
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*image_size, 3)
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input([*IMAGE_SIZE, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")


def get_mean_std(x):
    """Computes the mean and standard deviation of a tensor."""
    axes = [1, 2]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + 1e-5)
    return mean, standard_deviation


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


class AdaINModel(keras.Model):
    def __init__(self, image_size=(None, None), style_weight=4.0, **kwargs):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.encoder = get_mini_vgg19_encoder(image_size)
        self.decoder = get_adain_decoder()
        self.adain_layer = AdaIN()
        self.loss_net = get_loss_net(image_size)
        self.style_weight = style_weight

    def produce_functional_model():
        style_input = layers.Input([*self.image_size, 3])
        style_encoded = self.encoder(style_input)
        content_input = layers.Input([*self.image_size, 3])
        content_encoded = self.encoder(content_input)

        adain_feature_map = self.adain_layer((style_encoded, content_encoded))
        content_decoded = self.decoder(adain_feature_map)
        
        return keras.Model([style_input, content_input], content_decoded, name="adain_style_transfer")


    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
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
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            reconstructed_image, t = self(inputs)
            # Compute the losses.
            reconstructed_vgg_features = self.loss_net(reconstructed_image)
            style_vgg_features = self.loss_net(style)
            loss_content = self.loss_fn(t, reconstructed_vgg_features[-1])
            for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                    std_inp, std_out
                )
            loss_style = self.style_weight * loss_style
            total_loss = loss_content + loss_style

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
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        # Generate the neural style transferred image.
        reconstructed_image, t = self(inputs)

        # Compute the losses.
        recons_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_features[-1])
        for inp, out in zip(style_vgg_features, recons_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style
        total_loss = loss_content + loss_style

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]    

