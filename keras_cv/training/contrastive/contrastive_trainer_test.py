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
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss
from keras_cv.models import DenseNet121
from keras_cv.training import ContrastiveTrainer


class ContrastiveTrainerTest(tf.test.TestCase):
    def test_probe_requires_probe_optimizer(self):
        trainer = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=self.build_probe(),
        )
        with self.assertRaises(ValueError):
            trainer.compile(
                encoder_optimizer=optimizers.Adam(),
                encoder_loss=SimCLRLoss(temperature=0.5),
            )

    def test_targets_required_if_probing(self):
        trainer_with_probing = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=self.build_probe(),
        )
        trainer_without_probing = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=None,
        )

        images = tf.random.uniform((1, 50, 50, 3))

        trainer_with_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
            probe_optimizer=optimizers.Adam(),
            probe_loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        )
        trainer_without_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
        )

        with self.assertRaises(ValueError):
            trainer_with_probing.fit(images)

    def test_train_with_probing(self):
        trainer_with_probing = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=self.build_probe(classes=20),
        )

        images = tf.random.uniform((1, 50, 50, 3))
        targets = tf.ones((1, 20))

        trainer_with_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
            probe_metrics=[metrics.TopKCategoricalAccuracy(3, "top3_probe_accuracy")],
            probe_optimizer=optimizers.Adam(),
            probe_loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        )

        trainer_with_probing.fit(images, targets)

    def test_train_without_probing(self):
        trainer_without_probing = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=None,
        )

        images = tf.random.uniform((1, 50, 50, 3))
        targets = tf.ones((1, 20))

        trainer_without_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
        )

        trainer_without_probing.fit(images)
        trainer_without_probing.fit(images, targets)

    def test_inference_not_supported(self):
        trainer = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=self.build_augmenter(),
            projector=self.build_projector(),
            probe=None,
        )
        trainer.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
        )

        with self.assertRaises(NotImplementedError):
            trainer(tf.ones((1, 50, 50, 3)))

    def test_encoder_must_have_flat_output(self):
        with self.assertRaises(ValueError):
            _ = ContrastiveTrainer(
                # A DenseNet without pooling does not have a flat output
                encoder=DenseNet121(include_rescaling=False, include_top=False),
                augmenter=self.build_augmenter(),
                projector=self.build_projector(),
                probe=None,
            )

    def test_with_multiple_augmenters_and_projectors(self):
        augmenter0 = preprocessing.RandomFlip("horizontal")
        augmenter1 = preprocessing.RandomFlip("vertical")

        projector0 = layers.Dense(64, name="projector0")
        projector1 = keras.Sequential(
            [projector0, layers.ReLU(), layers.Dense(64, name="projector1")]
        )

        trainer_without_probing = ContrastiveTrainer(
            encoder=self.build_encoder(),
            augmenter=(augmenter0, augmenter1),
            projector=(projector0, projector1),
            probe=None,
        )

        images = tf.random.uniform((1, 50, 50, 3))

        trainer_without_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
        )

        trainer_without_probing.fit(images)

    def build_augmenter(self):
        return preprocessing.RandomFlip("horizontal")

    def build_encoder(self):
        return DenseNet121(include_rescaling=False, include_top=False, pooling="avg")

    def build_projector(self):
        return layers.Dense(128)

    def build_probe(self, classes=20):
        return layers.Dense(classes)
