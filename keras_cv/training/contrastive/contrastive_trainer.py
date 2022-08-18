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


class ContrastiveTrainer(keras.Model):
    """Creates a self-supervised contrastive trainer for a model.

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        augmenter: a preprocessing layer to randomly augment input images for contrastive learning.
        projector: a projection model for contrastive training
        include_probe: Whether to include a single fully-connected layer during
            training for probing classification accuracy using the learned encoding.
            Note that this should be specified iff training with labeled images.
            If provided, `classes` must be provided.
        probe_metrics: a list of metrics for the linear probe, only used if `include_probe` is True.
        classes: optional number of classes to classify images into, only to be
            specified if `include_probe` is True.

    Returns:
      A `keras.Model` instance.

    """

    def __init__(
        self,
        encoder,
        augmenter,
        projector,
        include_probe,
        probe_metrics=[keras.metrics.CategoricalAccuracy(name="probe_accuracy")],
        classes=None,
    ):
        super().__init__()

        if encoder.output.shape.rank != 2:
            raise ValueError("Encoder must have a flattened output")

        if include_probe:
            if not classes:
                raise ValueError(
                    "`classes` must be specified when `include_probe` is `True`."
                )

        self.include_probe = include_probe

        self.augmenter = augmenter
        self.encoder = encoder
        self.projector = projector

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.probe_loss_metric = keras.metrics.Mean(name="probe_loss")
        self.probe_metrics = probe_metrics or []

        if self.include_probe:
            self.probing_top = layers.Dense(classes, name="linear_probe")

    def compile(self, optimizer, loss, probe_optimizer=None, **kwargs):
        super().compile(**kwargs)

        if self.include_probe and not probe_optimizer:
            raise ValueError(
                "`probe_optimizer` must be specified when `include_probe` is `True`."
            )

        self.optimizer = optimizer
        self.loss = loss

        if self.include_probe:
            self.probe_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
            self.probe_optimizer = probe_optimizer

    @property
    def metrics(self):
        metrics = [
            self.loss_metric,
        ]
        if self.include_probe:
            metrics += [self.probe_loss_metric]
            metrics += self.probe_metrics
        return super().metrics + metrics

    def train_step(self, data):
        if self.include_probe:
            if type(data) is not tuple or len(data) != 2:
                raise ValueError(
                    "Targets must be provided when `include_probe` is True"
                )
            images, labels = data
        else:
            if type(data) is tuple:
                raise ValueError(
                    "Targets must not be provided when `include_probe` is False"
                )
            images = data

        augmented_images_1 = self.augmenter(images, training=True)
        augmented_images_2 = self.augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projector(features_1, training=True)
            projections_2 = self.projector(features_2, training=True)

            loss = self.loss(projections_1, projections_2)

        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights + self.projector.trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projector.trainable_weights,
            )
        )
        self.loss_metric.update_state(loss)

        if self.include_probe:
            with tf.GradientTape() as tape:
                features = self.encoder(images, training=False)
                class_logits = self.probing_top(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.probing_top.trainable_weights)
            self.probe_optimizer.apply_gradients(
                zip(gradients, self.probing_top.trainable_weights)
            )
            self.probe_loss_metric.update_state(probe_loss)
            for metric in self.probe_metrics:
                metric.update_state(labels, class_logits)

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, inputs):
        raise NotImplementedError(
            "ContrastiveTrainer.call() is not implemented - please call your model directly."
        )
