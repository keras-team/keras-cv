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
from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss


class SimCLR(keras.Model):
    def __init__(
        self,
        encoder,
        include_probing,
        classes=None,
        projection_width=128,
        augmenter=None,
        include_rescaling=None,
    ):
        super().__init__()

        self.include_probing = include_probing
        self.projection_width = projection_width

        if not augmenter and not include_rescaling:
            raise ValueError(
                "`include_rescaling` is required when using the default augmenter."
            )
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

        self.projection_top = keras.Sequential(
            [
                keras.Input(shape=(self.encoder.output.shape[-1],)),
                layers.Dense(self.projection_width, activation="relu"),
                layers.Dense(self.projection_width),
            ],
            name="projection_top",
        )

        if self.include_probing:
            if not classes:
                raise ValueError(
                    "`classes` must be specified when `include_probing` is `True`."
                )
            self.probing_top = keras.Sequential(
                [
                    layers.Input(shape=(self.encoder.output.shape[-1],)),
                    layers.Dense(classes),
                ],
                name="linear_probe",
            )

    def compile(
        self, contrastive_optimizer, probe_optimizer=None, temperature=0.1, **kwargs
    ):
        super().compile(**kwargs)

        # We call the contrastive optimizer `optimizer` so that Keras components
        # such as the ReduceLROnPlateau callback can correctly update this
        # optimizer.
        self.optimizer = contrastive_optimizer
        self.simclr_loss = SimCLRLoss(temperature)
        self.simclr_loss_metric = keras.metrics.Mean(name="simclr_loss")

        if self.include_probing:
            self.probe_loss = keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
            self.probe_loss_metric = keras.metrics.Mean(name="probe_loss")
            self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy(
                name="probe_accuracy"
            )

            if not probe_optimizer:
                raise ValueError(
                    "`probe_optimizer` must be specified when `include_probing` is `True`."
                )
            self.probe_optimizer = probe_optimizer

    @property
    def metrics(self):
        metrics = [
            self.simclr_loss_metric,
        ]
        if self.include_probing:
            metrics += [
                self.probe_loss_metric,
                self.probe_accuracy,
            ]
        return metrics

    def train_step(self, data):
        if self.include_probing:
            if type(data) is not tuple or len(data) != 2:
                raise ValueError(
                    "Targets must be provided when `include_probing` is True"
                )
            images, labels = data
        else:
            if type(data) is tuple:
                raise ValueError(
                    "Targets must not be provided when `include_probing` is False"
                )
            images = data

        augmented_images_1 = self.augmenter(images, training=True)
        augmented_images_2 = self.augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projection_top(features_1, training=True)
            projections_2 = self.projection_top(features_2, training=True)

            simclr_loss = self.simclr_loss.call(projections_1, projections_2)

        gradients = tape.gradient(
            simclr_loss,
            self.encoder.trainable_weights + self.projection_top.trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_top.trainable_weights,
            )
        )
        self.simclr_loss_metric.update_state(simclr_loss)

        if self.include_probing:
            with tf.GradientTape() as tape:
                features = self.encoder(images, training=False)
                class_logits = self.probing_top(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.probing_top.trainable_weights)
            self.probe_optimizer.apply_gradients(
                zip(gradients, self.probing_top.trainable_weights)
            )
            self.probe_loss_metric.update_state(probe_loss)
            self.probe_accuracy.update_state(labels, class_logits)

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, inputs):
        raise NotImplementedError("SimCLR models cannot be used for inference")
