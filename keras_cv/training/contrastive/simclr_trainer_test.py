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
from keras_cv.models import ResNet50V2
from keras_cv.training import SimCLRAugmenter
from keras_cv.training import SimCLRTrainer


class SimCLRTrainerTest(tf.test.TestCase):
    def test_train_without_probing(self):
        simclr_without_probing = SimCLRTrainer(
            self.build_encoder(),
            augmenter=SimCLRAugmenter(value_range=(0, 255)),
        )

        images = tf.random.uniform((10, 512, 512, 3))

        simclr_without_probing.compile(
            encoder_optimizer=optimizers.Adam(),
            encoder_loss=SimCLRLoss(temperature=0.5),
        )
        simclr_without_probing.fit(images)

    def build_encoder(self):
        return ResNet50V2(include_rescaling=False, include_top=False, pooling="avg")
