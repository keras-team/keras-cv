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
import json
import os

import tensorflow as tf
import keras_cv
from absl import app
from absl import flags
from keras.callbacks import BackupAndRestore
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from utils import get_learning_rate_schedule
from utils import load_cfar10_dataset

from keras_cv.models import DenseNet121

_GCS_BUCKET = flags.DEFINE_string("gcs_bucket", None, "Name of GCS Bucket")
_EXPERIMENT_ID = flags.DEFINE_string(
    "experiment_id", None, "An experiment name (preferably a git commit hash)"
)

NUM_CLASSES = 10
EPOCHS = 250
BATCH_SIZE = 32
WEIGHTS_PATH = "weights.hdf5"

AUGMENT_LAYERS = [
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.7),
    keras_cv.layers.CutMix(),
    keras_cv.layers.MixUp(),
    keras_cv.layers.RandomFlip(),
]

@tf.function
def augment(img, label):
    inputs = {"images": img, "labels": label}
    for layer in AUGMENT_LAYERS:
        inputs = layer(inputs)
    return inputs["images"], inputs["labels"]


def main(argv):
    assert _GCS_BUCKET.value
    assert _EXPERIMENT_ID.value

    gcs_path_base = "gs://{bucket}/densenet/{experiment}/".format(
        bucket=_GCS_BUCKET.value, experiment=_EXPERIMENT_ID.value
    )
    gcs_backup_path = gcs_path_base + "backup/"
    gcs_weights_path = gcs_path_base + WEIGHTS_PATH
    gcs_tensorboard_path = gcs_path_base + "logs/"

    train, test = load_cfar10_dataset(BATCH_SIZE)
    train = train.map(augment)

    gcs_backup = BackupAndRestore(gcs_backup_path)
    local_checkpoint = ModelCheckpoint(
        WEIGHTS_PATH,
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )
    tensorboard = TensorBoard(log_dir=gcs_tensorboard_path)

    with tf.distribute.MirroredStrategy().scope():
        model = DenseNet121(
            include_rescaling=True,
            include_top=True,
            num_classes=NUM_CLASSES,
            input_shape=(32, 32, 3),
        )

        model.compile(
            optimizer=Adam(
                learning_rate=get_learning_rate_schedule(
                    EPOCHS, steps_per_epoch=train.cardinality().numpy()
                )
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[gcs_backup, local_checkpoint, tensorboard],
            validation_data=test,
        )

    # In order to save only weights to GCS, we manually store weights locally
    # (in our local_checkpoint callback) and then copy them to GCS using gsutil.
    # In case storing weights in GCS fails, the weights are also stored locally.
    os.system(
        "gsutil cp {local_path} {remote_path}".format(
            local_path=WEIGHTS_PATH, remote_path=gcs_weights_path
        )
    )

    metadata = {
        "experiment_id": _EXPERIMENT_ID.value,
        "gcs_weights_path": gcs_weights_path,
        "gcs_tensorboard_path": gcs_tensorboard_path,
    }
    with open("densenet.json", "w") as outfile:
        json.dump(metadata, outfile)


if __name__ == "__main__":
    app.run(main)
