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
"""An example of transfer learning classification using KerasCV.
In this example, we use a DenseNet121 pre-trained against ImageNet.
We add a top to the DenseNet to learn cats vs dogs dataset.

In this example, we demonstrate that with minimal customization and pre-trained
KerasCV models, we can achieve very strong validation results on a new dataset.
"""
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers

from keras_cv import models

NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20


# Loading the TFDS dataset with batching, resizing, and one-hot encoding.
def load_cats_vs_dogs():
    train_ds, test_ds = tfds.load(
        "cats_vs_dogs", split=["train[:90%]", "train[90%:]"], as_supervised=True
    )
    resizing = layers.Resizing(150, 150)
    train = train_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(BATCH_SIZE)
    test = test_ds.map(
        lambda x, y: (resizing(x), tf.one_hot(y, 2)),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).batch(BATCH_SIZE)
    return train, test


# Building a model using a pre-trained DenseNet and a single dense layer top.
densenet = models.DenseNet121(
    include_rescaling=True,
    include_top=False,
    weights="imagenet/classification",
    pooling="avg",
)
densenet.trainable = False
model = keras.models.Sequential(
    [densenet, layers.Dense(NUM_CLASSES, activation="softmax")]
)

train, test = load_cats_vs_dogs()
train, test = train.prefetch(tf.data.AUTOTUNE), test.prefetch(tf.data.AUTOTUNE)

# Compile and fit the model with all default settings.
# This scores 97% validation accuracy after just 1 epoch!
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=test, batch_size=BATCH_SIZE, epochs=EPOCHS)
