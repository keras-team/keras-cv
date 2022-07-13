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
from utils import build_checkpoint_callback
from utils import load_cfar10_dataset

from keras_cv.models import DenseNet121

train, test = load_cfar10_dataset()

NUM_CLASSES = 10
EPOCHS = 1
WEIGHTS_PATH = "weights.hdf5"
MODEL_GS_PATH = "densenet/"


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

    if tf.io.gfile.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)

    model.fit(
        train,
        batch_size=32,
        epochs=EPOCHS,
        callbacks=[build_checkpoint_callback(WEIGHTS_PATH)],
        validation_data=test,
    )

model.save("gs://ian-kerascv/" + MODEL_GS_PATH + WEIGHTS_PATH)
