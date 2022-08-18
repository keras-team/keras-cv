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
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss
from keras_cv.models import DenseNet121
from keras_cv.training import ContrastiveTrainer


class ContrastiveTrainerTest(tf.test.TestCase):
    def test_include_probe_requires_classes(self):
        with self.assertRaises(ValueError):
            _ = ContrastiveTrainer(
                self.build_encoder(),
                self.build_augmenter(),
                self.build_projector(),
                include_probe=True,
                classes=None,
            )

    def test_include_probe_requires_probe_optimizer(self):
        trainer = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=10,
        )
        with self.assertRaises(ValueError):
            trainer.compile(optimizers.Adam(), SimCLRLoss(temperature=0.5))

    def test_targets_required_iff_probing(self):
        trainer_with_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=20,
        )
        trainer_without_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )

        images = tf.random.uniform((10, 512, 512, 3))
        targets = tf.ones((10, 20))

        trainer_with_probing.compile(
            optimizers.Adam(),
            loss=SimCLRLoss(temperature=0.5),
            probe_optimizer=optimizers.Adam(),
        )
        trainer_without_probing.compile(
            optimizers.Adam(), loss=SimCLRLoss(temperature=0.5)
        )

        with self.assertRaises(ValueError):
            trainer_with_probing.fit(images)
        with self.assertRaises(ValueError):
            trainer_without_probing.fit(images, targets)

    def test_train_with_probing(self):
        trainer_with_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=20,
        )

        images = tf.random.uniform((10, 512, 512, 3))
        targets = tf.ones((10, 20))

        trainer_with_probing.compile(
            optimizers.Adam(),
            loss=SimCLRLoss(temperature=0.5),
            probe_metrics=[metrics.TopKCategoricalAccuracy(3, "top3_probe_accuracy")],
            probe_optimizer=optimizers.Adam(),
        )

        trainer_with_probing.fit(images, targets)

    def test_train_without_probing(self):
        trainer_without_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )

        images = tf.random.uniform((10, 512, 512, 3))

        trainer_without_probing.compile(
            optimizers.Adam(), loss=SimCLRLoss(temperature=0.5)
        )

        trainer_without_probing.fit(images)

    def test_inference_not_supported(self):
        trainer = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )
        trainer.compile(optimizer=optimizers.Adam(), loss=SimCLRLoss(temperature=0.5))

        with self.assertRaises(NotImplementedError):
            trainer(tf.ones((10, 250, 250, 3)))

    def test_encoder_must_have_flat_output(self):
        with self.assertRaises(ValueError):
            _ = ContrastiveTrainer(
                # A DenseNet without pooling does not have a flat output
                DenseNet121(include_rescaling=False, include_top=False),
                self.build_augmenter(),
                self.build_projector(),
                include_probe=False,
            )

    def build_augmenter(self):
        return preprocessing.RandomFlip("horizontal")

    def build_encoder(self):
        return DenseNet121(include_rescaling=False, include_top=False, pooling="avg")

    def build_projector(self):
        return layers.Dense(128)
