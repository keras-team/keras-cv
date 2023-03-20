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
Title: Benchmarking a KerasCV model against ImageNetV2
Author: [DavidLandup0](https://github.com/DavidLandup0)
Date created: 2022/12/14
Last modified: 2022/12/14
Description: Use KerasCV architectures and benchmark them against ImageNetV2 from TensorFlow Datasets
"""

import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from tensorflow import keras

from keras_cv import models

flags.DEFINE_string(
    "model_name", None, "The name of the model in KerasCV.models to use."
)
flags.DEFINE_boolean(
    "include_rescaling",
    True,
    "Whether to include rescaling or not at the start of the model.",
)
flags.DEFINE_string(
    "model_kwargs",
    "{}",
    "Keyword argument dictionary to pass to the constructor of the model being evaluated.",
)

flags.DEFINE_integer(
    "batch_size",
    32,
    "The batch size for the evaluation set.",
)

flags.DEFINE_string(
    "weights",
    "imagenet",
    "The path to the weights to load for the model.",
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

model = models.__dict__[FLAGS.model_name]
model = model(
    include_rescaling=FLAGS.include_rescaling,
    include_top=True,
    num_classes=1000,
    input_shape=(224, 224, 3),
    weights=FLAGS.weights,
    **eval(FLAGS.model_kwargs),
)

model.compile(
    "adam",
    "sparse_categorical_crossentropy",
    metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(5)],
)


def preprocess_image(img, label):
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    return img, label


# Todo
# Include imagenet_val and imagenet_real as well and report
# results for all three
(test_set), info = tfds.load(
    "imagenet_v2", split=["test"], as_supervised=True, with_info=True
)
test_set = (
    test_set[0]
    .shuffle(len(test_set))
    .map(preprocess_image)
    .batch(FLAGS.batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Todo
# Create a nicer report, include inference time
# model size, etc.
loss, acc, top_5 = model.evaluate(test_set, verbose=0)
print(
    f"Benchmark results:\n{'='*25}\n{FLAGS.model_name} achieves: \n - Top-1 Accuracy: {acc*100} \n - Top-5 Accuracy: {top_5*100} \non ImageNetV2 with setup:"
)
print(
    f"- model_name: {FLAGS.model_name}\n"
    f"- include_rescaling: {FLAGS.include_rescaling}\n"
    f"- batch_size: {FLAGS.batch_size}\n"
    f"- weights: {FLAGS.weights}\n"
    f"- model_kwargs: {FLAGS.model_kwargs}\n"
)
