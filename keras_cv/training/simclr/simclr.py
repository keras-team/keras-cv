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

# Adapted from https://keras.io/examples/vision/semisupervised_simclr/#introduction
# This is a work-in-progress

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss


class SimCLR(keras.Model):
    def __init__(
        self,
        encoder,
        include_rescaling,
        projection_head_width=128,
        augmenter=None,
    ):
        super().__init__()

        self.projection_head_width = projection_head_width

        self.augmenter = augmenter or preprocessing.Augmenter(
            [
                preprocessing.RandomFlip("horizontal"),
                preprocessing.RandomTranslation(0.25, 0.25),
                preprocessing.RandomZoom((-0.5, 0.0), (-0.5, 0.0)),
                preprocessing.RandomColorJitter(
                    value_range=[0, 255] if include_rescaling else [0, 1],
                    brightness_factor=0.5,
                    contrast_factor=0.5,
                    saturation_factor=(0.3, 0.7),
                    hue_factor=0.5,
                ),
            ]
        )

        self.encoder = encoder

        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(self.encoder.output.shape[-1],)),
                layers.Dense(self.projection_head_width, activation="relu"),
                layers.Dense(self.projection_head_width),
            ],
            name="projection_head",
        )

    def compile(self, contrastive_optimizer, temperature=0.1, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.simclr_loss = SimCLRLoss(temperature)

        self.simclr_loss_metric = keras.metrics.Mean(name="simclr_loss")

    @property
    def metrics(self):
        return [
            self.simclr_loss_metric,
        ]

    def train_step(self, data):
        images = data

        augmented_images_1 = self.augmenter(images, training=True)
        augmented_images_2 = self.augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            simclr_loss = self.simclr_loss.call(projections_1, projections_2)

        gradients = tape.gradient(
            simclr_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.simclr_loss_metric.update_state(simclr_loss)

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, inputs):
        raise NotImplementedError("SimCLR models cannot be used for inference")


# Testing code
# from keras_cv.models import DenseNet121
# encoder = DenseNet121(include_rescaling=True, include_top=False, pooling='avg')
# simclr = SimCLR(encoder, include_rescaling=True)
# simclr.compile(keras.optimizers.Adam())
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# simclr.fit(x_train[:500])
