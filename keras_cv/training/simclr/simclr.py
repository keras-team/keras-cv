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


class SimCLR(keras.Model):
    def __init__(
        self,
        model,
        include_rescaling,
        input_shape=(None, None, 3),
        temperature=0.1,
        projection_head_width=128,
        augmenter=None,
    ):
        super().__init__()

        self.temperature = temperature
        self.projection_head_width = projection_head_width

        self.augmenter = augmenter or keras.Sequential(
            [
                keras.Input(shape=input_shape),
                preprocessing.RandomFlip("horizontal"),
                layers.RandomTranslation(0.25, 0.25),
                layers.RandomZoom((-0.5, 0.0), (-0.5, 0.0)),
                preprocessing.RandomColorJitter(
                    value_range=[0, 255] if include_rescaling else [0, 1],
                    brightness_factor=0.5,
                    contrast_factor=0.5,
                    saturation_factor=(0.3, 0.7),
                    hue_factor=0.5,
                ),
            ]
        )

        self.encoder = model(
            include_rescaling=include_rescaling, include_top=False, pooling="avg"
        )

        self.projection_head = keras.Sequential(
            [
                keras.Input(shape=(self.encoder.output.shape[-1],)),
                layers.Dense(self.projection_head_width, activation="relu"),
                layers.Dense(self.projection_head_width),
            ],
            name="projection_head",
        )

    def compile(self, contrastive_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer

        self.contrastive_loss_metric = keras.metrics.Mean(name="contrastive_loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="contrastive_accuracy"
        )

    @property
    def metrics(self):
        return [
            self.contrastive_loss_metric,
            self.contrastive_accuracy,
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def train_step(self, data):
        images = data

        augmented_images_1 = self.augmenter(images, training=True)
        augmented_images_2 = self.augmenter(images, training=True)

        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)

            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)

            contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )

        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.contrastive_loss_metric.update_state(contrastive_loss)

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        return {}
