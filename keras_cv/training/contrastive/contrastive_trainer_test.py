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
from tensorflow.keras import optimizers

from keras_cv.layers import preprocessing
from keras_cv.losses import SimCLRLoss
from keras_cv.models import DenseNet121
from keras_cv.training import ContrastiveTrainer
from keras_cv.training import SimCLRTrainer


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
        simclr = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=10,
        )
        with self.assertRaises(ValueError):
            simclr.compile(optimizers.Adam(), SimCLRLoss(temperature=0.5))

    def test_targets_required_iff_probing(self):
        simclr_with_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=20,
        )
        simclr_without_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )

        images = tf.random.uniform((10, 512, 512, 3))
        targets = tf.ones((10, 20))

        simclr_with_probing.compile(
            optimizers.Adam(),
            loss=SimCLRLoss(temperature=0.5),
            probe_optimizer=optimizers.Adam(),
        )
        simclr_without_probing.compile(
            optimizers.Adam(), loss=SimCLRLoss(temperature=0.5)
        )

        with self.assertRaises(ValueError):
            simclr_with_probing.fit(images)
        with self.assertRaises(ValueError):
            simclr_without_probing.fit(images, targets)

    def test_train_with_probing(self):
        simclr_with_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=True,
            classes=20,
        )

        images = tf.random.uniform((10, 512, 512, 3))
        targets = tf.ones((10, 20))

        simclr_with_probing.compile(
            optimizers.Adam(),
            loss=SimCLRLoss(temperature=0.5),
            probe_optimizer=optimizers.Adam(),
        )

        simclr_with_probing.fit(images, targets)

    def test_train_without_probing(self):
        simclr_without_probing = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )

        images = tf.random.uniform((10, 512, 512, 3))

        simclr_without_probing.compile(
            optimizers.Adam(), loss=SimCLRLoss(temperature=0.5)
        )

        simclr_without_probing.fit(images)

    def test_inference_not_supported(self):
        simclr = ContrastiveTrainer(
            self.build_encoder(),
            self.build_augmenter(),
            self.build_projector(),
            include_probe=False,
        )
        simclr.compile(optimizer=optimizers.Adam(), loss=SimCLRLoss(temperature=0.5))

        with self.assertRaises(NotImplementedError):
            simclr(tf.ones((10, 250, 250, 3)))

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
