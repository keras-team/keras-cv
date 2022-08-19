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


class ContrastiveTrainer(keras.Model):
    """Creates a self-supervised contrastive trainer for a model.

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        augmenter: a preprocessing layer to randomly augment input images for contrastive learning,
            or a tuple of two separate augmenters for the two sides of the contrastive pipeline.
        projector: a projection model for contrastive training, or a tuple of two separate
            projectors for the two sides of the contrastive pipeline.
        include_probe: Whether to include a single fully-connected layer during
            training for probing classification accuracy using the learned encoding.
            Note that this should be specified iff training with labeled images.
            If provided, `classes` must be provided.
        probe_metrics: a list of metrics for the linear probe, only used if `include_probe` is True.
        classes: optional number of classes to classify images into, only to be
            specified if `include_probe` is True.

    Returns:
      A `keras.Model` instance.


    Usage:
    ```python
    encoder = keras_cv.models.DenseNet121(include_rescaling=False, include_top=False, pooling="avg")
    augmenter = keras_cv.layers.preprocessing.RandomFlip()
    projector = keras.layers.Dense(64)

    trainer = keras_cv.training.ContrastiveTrainer(
        encoder,
        augmenter,
        projector
    )

    trainer.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras_cv.losses.SimCLRLoss()
    )

    unlabeled_images = load_data()
    trainer.fit(unlabeled_images)
    ```

    """

    def __init__(
        self,
        encoder,
        augmenter,
        projector,
        include_probe,
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

        if type(augmenter) is tuple and len(augmenter) != 2:
            raise ValueError(
                "`augmenter` must be either a single augmenter or a tuple of exactly 2 augmenters."
            )

        if type(projector) is tuple and len(projector) != 2:
            raise ValueError(
                "`augmenter` must be either a single augmenter or a tuple of exactly 2 augmenters."
            )

        self.augmenters = (
            augmenter if type(augmenter) is tuple else (augmenter, augmenter)
        )
        self.encoder = encoder
        self.projectors = (
            projector if type(projector) is tuple else (projector, projector)
        )

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.probe_loss_metric = keras.metrics.Mean(name="probe_loss")
        self.probe_metrics = []

        self.include_probe = include_probe

        if self.include_probe:
            self.probing_top = layers.Dense(classes, name="linear_probe")

    def compile(
        self,
        optimizer,
        loss,
        probe_optimizer=None,
        probe_loss=None,
        probe_metrics=None,
        **kwargs
    ):
        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        if self.include_probe and not probe_optimizer:
            raise ValueError(
                "`probe_optimizer` must be specified when `include_probe` is `True`."
            )

        if self.include_probe and not probe_loss:
            raise ValueError(
                "`probe_loss` must be specified when `include_probe` is `True`."
            )

        if self.include_probe:
            self.probe_loss = probe_loss
            self.probe_optimizer = probe_optimizer
            self.probe_metrics = probe_metrics or []

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

        augmented_images_0 = self.augmenters[0](images, training=True)
        augmented_images_1 = self.augmenters[1](images, training=True)

        with tf.GradientTape() as tape:
            features_0 = self.encoder(augmented_images_0, training=True)
            features_1 = self.encoder(augmented_images_1, training=True)

            projections_0 = self.projectors[0](features_0, training=True)
            projections_1 = self.projectors[1](features_1, training=True)

            # TODO(ianstenbit), add regularization_losses from encoder and projectors
            loss = self.compiled_loss(projections_0, projections_1)

        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights
            + self.projectors[0].trainable_weights
            + self.projectors[1].trainable_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights
                + self.projectors[0].trainable_weights
                + self.projectors[1].trainable_weights,
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
