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
from keras.layers import Resizing
from tensorflow.keras.optimizers.schedules import PolynomialDecay


def load_cats_and_dogs_dataset(batch_size=32):
    train_ds, test_ds = tfds.load(
        "cats_vs_dogs", split=["train[:90%]", "train[90%:]"], as_supervised=True
    )

    resizing = Resizing(150, 150)

    train = train_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)
    test = test_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(batch_size)

    return train, test


def load_cifar10_dataset(batch_size=32):
    train_ds, test_ds = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True
    )

    train = train_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(batch_size)
    test = test_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(batch_size)

    return train, test


def get_learning_rate_schedule(decay_steps):
    return PolynomialDecay(
        initial_learning_rate=0.01, decay_steps=decay_steps, end_learning_rate=0.0001
    )
