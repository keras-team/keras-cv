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
"""Utility functions for training demos."""

import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay


def load_cfar10_dataset(batch_size=32):
    train_ds, test_ds = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True
    )

    train = train_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(batch_size)
    test = test_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(batch_size)

    return train, test


def get_learning_rate_schedule(epochs, steps_per_epoch):
    epoch_boundaries = [epochs / 20, epochs / 10, epochs / 5, epochs / 2]
    values = [0.01, 0.005, 0.001, 0.0005, 0.00025]

    boundaries = [steps_per_epoch * x for x in epoch_boundaries]

    return PiecewiseConstantDecay(boundaries, values)
