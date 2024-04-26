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

from keras_cv.src.utils.train import convert_inputs_to_tf_dataset


class ContrastiveTrainer(keras.Model):
    """Creates a self-supervised contrastive trainer for a model.

    Args:
        encoder: a `keras.Model` to be pre-trained. In most cases, this encoder
            should not include a top dense layer.
        augmenter: a preprocessing layer to randomly augment input images for
            contrastive learning, or a tuple of two separate augmenters for the
            two sides of the contrastive pipeline.
        projector: a projection model for contrastive training, or a tuple of
            two separate projectors for the two sides of the contrastive
            pipeline. This shrinks the feature map produced by the encoder, and
            is usually a 1 or 2-layer dense MLP.
        probe: An optional Keras layer or model which will be trained against
            class labels at train-time using the encoder output as input.
            Note that this should be specified iff training with labeled images.
            This predicts class labels based on the feature map produced by the
            encoder and is usually a 1 or 2-layer dense MLP.

    Returns:
      A `keras.Model` instance.


    Example:
    ```python
    encoder = keras.Sequential(
        [
            DenseNet121Backbone(include_rescaling=False),
            layers.GlobalAveragePooling2D(name="avg_pool"),
        ],
    )
    augmenter = keras_cv.layers.preprocessing.RandomFlip()
    projector = keras.layers.Dense(64)
    probe = keras_cv.training.ContrastiveTrainer.linear_probe(num_classes=10)

    trainer = keras_cv.training.ContrastiveTrainer(
        encoder=encoder,
        augmenter=augmenter,
        projector=projector,
        probe=probe
    )

    trainer.compile(
        encoder_optimizer=keras.optimizers.Adam(),
        encoder_loss=keras_cv.losses.SimCLRLoss(temperature=0.5),
        probe_optimizer=keras.optimizers.Adam(),
        probe_loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        probe_metrics=[keras.metrics.CategoricalAccuracy(name="probe_accuracy")]
    )

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, 10)

    trainer.fit(x_train, y_train)
    ```

    """

    def __init__(
        self,
        encoder,
        augmenter,
        projector,
        probe=None,
    ):
        super().__init__()

        if encoder.output.shape.rank != 2:
            raise ValueError(
                f"`encoder` must have a flattened output. Expected "
                f"rank(encoder.output.shape)=2, got "
                f"encoder.output.shape={encoder.output.shape}"
            )

        if type(augmenter) is tuple and len(augmenter) != 2:
            raise ValueError(
                "`augmenter` must be either a single augmenter or a tuple of "
                "exactly 2 augmenters."
            )

        if type(projector) is tuple and len(projector) != 2:
            raise ValueError(
                "`projector` must be either a single augmenter or a tuple of "
                "exactly 2 augmenters."
            )

        self.augmenters = (
            augmenter if type(augmenter) is tuple else (augmenter, augmenter)
        )
        self.encoder = encoder
        # Check to see if the projector is being shared or are distinct.
        self._is_shared_projector = (
            True if not isinstance(projector, tuple) else False
        )
        self.projectors = (
            projector if type(projector) is tuple else (projector, projector)
        )
        self.probe = probe

        self.loss_metric = keras.metrics.Mean(name="loss")

        if probe is not None:
            self.probe_loss_metric = keras.metrics.Mean(name="probe_loss")
            self.probe_metrics = []

    def compile(
        self,
        encoder_loss,
        encoder_optimizer,
        encoder_metrics=None,
        probe_optimizer=None,
        probe_loss=None,
        probe_metrics=None,
        **kwargs,
    ):
        super().compile(
            loss=encoder_loss,
            optimizer=encoder_optimizer,
            metrics=encoder_metrics,
            **kwargs,
        )

        if self.probe and not probe_optimizer:
            raise ValueError(
                "`probe_optimizer` must be specified when a probe is included."
            )

        if self.probe and not probe_loss:
            raise ValueError(
                "`probe_loss` must be specified when a probe is included."
            )

        if "loss" in kwargs:
            raise ValueError(
                "`loss` parameter in ContrastiveTrainer.compile is ambiguous. "
                "Please specify `encoder_loss` or `probe_loss`."
            )

        if "optimizer" in kwargs:
            raise ValueError(
                "`optimizer` parameter in ContrastiveTrainer.compile is "
                "ambiguous. Please specify `encoder_optimizer` or "
                "`probe_optimizer`."
            )

        if "metrics" in kwargs:
            raise ValueError(
                "`metrics` parameter in ContrastiveTrainer.compile is "
                "ambiguous. Please specify `encoder_metrics` or "
                "`probe_metrics`."
            )

        if self.probe:
            self.probe_loss = probe_loss
            self.probe_optimizer = probe_optimizer
            self.probe_metrics = probe_metrics or []

    @property
    def metrics(self):
        metrics = [
            self.loss_metric,
        ]
        if self.probe:
            metrics += [self.probe_loss_metric]
            metrics += self.probe_metrics
        return super().metrics + metrics

    def fit(
        self,
        x=None,
        y=None,
        sample_weight=None,
        batch_size=None,
        **kwargs,
    ):
        dataset = convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
        )

        dataset = dataset.map(
            self.run_augmenters, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return super().fit(x=dataset, **kwargs)

    def run_augmenters(self, x, y=None):
        inputs = {"images": x}
        if y is not None:
            inputs["labels"] = y

        inputs["augmented_images_0"] = self.augmenters[0](x, training=True)
        inputs["augmented_images_1"] = self.augmenters[1](x, training=True)

        return inputs

    def train_step(self, data):
        images = data["images"]
        labels = data["labels"] if "labels" in data else None
        augmented_images_0 = data["augmented_images_0"]
        augmented_images_1 = data["augmented_images_1"]

        with tf.GradientTape() as tape:
            features_0 = self.encoder(augmented_images_0, training=True)
            features_1 = self.encoder(augmented_images_1, training=True)

            projections_0 = self.projectors[0](features_0, training=True)
            projections_1 = self.projectors[1](features_1, training=True)

            loss = self.compiled_loss(
                projections_0,
                projections_1,
                regularization_losses=self.encoder.losses,
            )

        # If the projector is shared, then take the trainable weights of just
        # one of the projectors in the tuple. If not, use both the projectors.
        projector_weights = (
            self.projectors[0].trainable_weights
            if self._is_shared_projector
            else self.projectors[0].trainable_weights
            + self.projectors[1].trainable_weights
        )
        gradients = tape.gradient(
            loss,
            self.encoder.trainable_weights + projector_weights,
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + projector_weights,
            )
        )
        self.loss_metric.update_state(loss)

        if self.probe:
            if labels is None:
                raise ValueError(
                    "Targets must be provided when a probe is specified"
                )
            with tf.GradientTape() as tape:
                features = tf.stop_gradient(
                    self.encoder(images, training=False)
                )
                class_logits = self.probe(features, training=True)
                probe_loss = self.probe_loss(labels, class_logits)
            gradients = tape.gradient(probe_loss, self.probe.trainable_weights)
            self.probe_optimizer.apply_gradients(
                zip(gradients, self.probe.trainable_weights)
            )
            self.probe_loss_metric.update_state(probe_loss)
            for metric in self.probe_metrics:
                metric.update_state(labels, class_logits)

        return {metric.name: metric.result() for metric in self.metrics}

    def call(self, inputs):
        raise NotImplementedError(
            "ContrastiveTrainer.call() is not implemented - "
            "please call your model directly."
        )

    @staticmethod
    def linear_probe(num_classes, **kwargs):
        return keras.Sequential(keras.layers.Dense(num_classes), **kwargs)
