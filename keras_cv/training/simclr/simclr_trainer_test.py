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
from tensorflow.keras import optimizers

from keras_cv.losses import SimCLRLoss
from keras_cv.models import DenseNet121
from keras_cv.training import SimCLRTrainer


class SimCLRTrainerTest(tf.test.TestCase):
    def test_default_augmenter_requires_value_range(self):
        with self.assertRaises(ValueError):
            _ = SimCLRTrainer(
                self.build_encoder(),
                include_probe=False,
                augmenter=None,
                value_range=None,
            )

    def test_include_probe_requires_classes(self):
        with self.assertRaises(ValueError):
            _ = SimCLRTrainer(
                self.build_encoder(),
                include_probe=True,
                classes=None,
                value_range=(0, 1),
            )

    def test_include_probe_requires_probe_optimizer(self):
        simclr = SimCLRTrainer(
            self.build_encoder(), include_probe=True, classes=10, value_range=(0, 1)
        )
        with self.assertRaises(ValueError):
            simclr.compile(optimizers.Adam(), SimCLRLoss(temperature=0.5))

    def test_targets_required_iff_probing(self):
        simclr_with_probing = SimCLRTrainer(
            self.build_encoder(), include_probe=True, classes=20, value_range=(0, 1)
        )
        simclr_without_probing = SimCLRTrainer(
            self.build_encoder(), include_probe=False, value_range=(0, 1)
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
        simclr_with_probing = SimCLRTrainer(
            self.build_encoder(), include_probe=True, classes=20, value_range=(0, 1)
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
        simclr_without_probing = SimCLRTrainer(
            self.build_encoder(), include_probe=False, value_range=(0, 1)
        )

        images = tf.random.uniform((10, 512, 512, 3))

        simclr_without_probing.compile(
            optimizers.Adam(), loss=SimCLRLoss(temperature=0.5)
        )

        simclr_without_probing.fit(images)

    def test_inference_not_supported(self):
        simclr = SimCLRTrainer(
            self.build_encoder(), include_probe=False, value_range=(0, 1)
        )
        simclr.compile(optimizer=optimizers.Adam(), loss=SimCLRLoss(temperature=0.5))

        with self.assertRaises(NotImplementedError):
            simclr(tf.ones((10, 250, 250, 3)))

    def test_encoder_must_have_flat_output(self):
        with self.assertRaises(ValueError):
            _ = SimCLRTrainer(
                # A DenseNet without pooling does not have a flat output
                DenseNet121(include_rescaling=False, include_top=False),
                include_probe=False,
                augmenter=None,
                value_range=None,
            )

    def build_encoder(self):
        return DenseNet121(include_rescaling=False, include_top=False, pooling="avg")
