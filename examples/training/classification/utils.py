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
from keras.callbacks import ModelCheckpoint, BackupAndRestore


def load_cfar10_dataset():
    train_ds, test_ds = tfds.load(
        "cifar10", split=["train", "test"], as_supervised=True
    )

    train = train_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(32)
    test = test_ds.map(lambda x, y: (x, tf.one_hot(y, 10))).batch(32)

    return train, test


def build_checkpoint_callback(weights_path):
    return ModelCheckpoint(
        weights_path,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )

def build_backup_and_restore_callback(backup_path):
    return BackupAndRestore(backup_path)
