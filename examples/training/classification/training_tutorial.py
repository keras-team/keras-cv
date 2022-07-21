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
"""
Title: Training a DenseNet for Image Classification with KerasCV
Author: [ianjjohnson](https://github.com/ianjjohnson)
Date created: 2022/07/21
Last modified: 2022/07/21
Description: Use KerasCV to train a DenseNet using modern best practives for image classification
"""

"""
## Overview
KerasCV makes training state-of-the-art classification models easy by providing implementations of modern models, preprocessing techniques, and layers. In this tutorial, we walk through training a DenseNet model against the cats and dogs dataset using Keras and KerasCV. Throughout this tutorial, we use some utility methods for data loading, etc. that can be found in KerasCV on [GitHub](github.com/keras-team/keras-cv).
"""

"""
## Imports & setup
This tutorial requires you to have KerasCV installed:
```shell
pip install keras-cv
```
We begin by importing all required packages:
"""
import json
import os

import tensorflow as tf
import wandb
from absl import app
from absl import flags
from keras.callbacks import BackupAndRestore
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from utils import augment
from utils import load_cats_and_dogs_dataset
from utils import save_training_results
from wandb.keras import WandbCallback

import keras_cv
from keras_cv.models import DenseNet121

"""
## Data loading
This guide uses the
[Cats vs Dogs dataset](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
for demonstration purposes.
To get started, we first load the dataset using the `load_cats_and_dogs_dataset` from our KerasCV training utils. Note that this method performs image resizing, one-hot encoding of targets, and batching for us.
"""

NUM_CLASSES = 2
BATCH_SIZE = 32
WIDTH = 150
HEIGHT = 150
EPOCHS = 250
train, test = load_cats_and_dogs_dataset(
    batch_size=BATCH_SIZE, width=WIDTH, height=HEIGHT
)


"""
Next, we augment our dataset. We define a set of augmentation layers and then apply them to our input dataset using the `apply_augmentation` method from our KerasCV training utils. Note that both before and after augmentation we can visualize our dataset using the `visualize_dataset` method from the KerasCV training utils.
"""

# (Optionally): visualize_dataset(train, "Pre-augmentation")

AUGMENT_LAYERS = [
    keras_cv.layers.RandomFlip(),
    keras_cv.layers.RandAugment(value_range=(0, 255), magnitude=0.3),
    keras_cv.layers.RandomCutout(height_factor=0.1, width_factor=0.1),
]
train = train.map(
    augment(AUGMENT_LAYERS), num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# (Optionally): visualize_dataset(train, "Post-augmentation")

"""
Now we can begin training our model. We begin by loading a DenseNet model from KerasCV
"""


def get_model():
    return DenseNet121(
        include_rescaling=True,
        include_top=True,
        num_classes=NUM_CLASSES,
        input_shape=(WIDTH, HEIGHT, 3),
    )


"""
Next, we pick an optimizer. Here we use Adam with a linearly decaying learning rate
"""


def get_optimizer():
    return Adam(
        learning_rate=PolynomialDecay(
            initial_learning_rate=0.005,
            decay_steps=train.cardinality().numpy() * EPOCHS,
            end_learning_rate=0.0001,
        )
    )


"""
Next, we pick a loss function. Here we use a built-in Keras loss function, so we simply specify it as a string.
"""


def get_loss_fn():
    return "binary_crossentropy"


"""
Next, we specify the metrics that we want to track. For this example, we track accuracy. Once again, accuracy is a built-in metric in Keras so we can specify it as a string.
"""


def get_metrics():
    return ["accuracy"]


"""
As a last piece of configuration, we configure callbacks for the method. We use EarlyStopping, BackupAndRestore, and a Weights and Biases (WandB) integration callback. Note that the WandB callback requires some initialization, so we do that in the `get_callbacks` method.
"""


def get_callbacks():
    wandb.init(project="densenet-training-tutorial", entity="keras-team-testing")
    return [
        EarlyStopping(patience=10),
        BackupAndRestore("training_backup/"),
        WandbCallback(),
    ]


"""
We can now compile the model and fit it to the training dataset.
"""

with tf.distribute.MirroredStrategy().scope():
    model = get_model()

    model.compile(
        optimizer=get_optimizer(),
        loss=get_loss_fn(),
        metrics=get_metrics(),
    )

    model.fit(
        train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        validation_data=test,
    )

"""
Next, we record validation metrics from the model and store metadata about the run as JSON using the `save_training_results` method from the KerasCV training utils.
"""
validation_metrics = model.evaluate(test, return_dict=True)

AUTHOR = "ianjjohnson"
RESULTS_PATH = "densenet_training_tutorial.json"
save_training_results(
    results_path=RESULTS_PATH, validation_metrics=validation_metrics, author=AUTHOR
)
