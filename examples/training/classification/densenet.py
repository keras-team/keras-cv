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
"""A training script for a DenseNet.

This example is under active development and should not be forked for other models yet.
"""
import tensorflow as tf
import os

from utils import build_backup_and_restore_callback, build_checkpoint_callback
from utils import load_cfar10_dataset

from keras_cv.models import DenseNet121

train, test = load_cfar10_dataset()

NUM_CLASSES = 10
EPOCHS = 10
PATH_BASE = "gs://ian-kerascv/densenet/experimentid/"
BACKUP_PATH = PATH_BASE + "backup/"
REMOTE_WEIGHTS_PATH = PATH_BASE + "weights.hdf5"
LOCAL_TMP_WEIGHTS_PATH = "tmp.hdf5"

backup = build_backup_and_restore_callback(BACKUP_PATH)

with tf.distribute.MirroredStrategy().scope():
    model = DenseNet121(
        include_rescaling=True,
        include_top=True,
        num_classes=NUM_CLASSES,
        input_shape=(32, 32, 3),
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train,
        batch_size=32,
        epochs=EPOCHS,
        callbacks=[backup],
        validation_data=test,
    )

model.save_weights(LOCAL_TMP_WEIGHTS_PATH)
os.system("gsutil -m cp " + LOCAL_TMP_WEIGHTS_PATH + " " + REMOTE_WEIGHTS_PATH)
os.system("rm " + LOCAL_TMP_WEIGHTS_PATH)

# TODO(ianjjohnson) After success, store a file with the final output (JSON)
