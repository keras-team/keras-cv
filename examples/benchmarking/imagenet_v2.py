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
Title: Benchmarking a KerasCV model against Imagenet Validation V2
Author: [DavidLandup0](https://github.com/DavidLandup0)
Date created: 2022/12/14
Last modified: 2022/12/14
Description: Use KerasCV architectures and benchmark them against ImageNetV2 from TensorFlow Datasets
"""

import sys

import tensorflow as tf
from absl import flags
from tensorflow import keras

import keras_cv
from keras_cv import models
import tensorflow_datasets as tfds

flags.DEFINE_string(
    "model_name", None, "The name of the model in KerasCV.models to use."
)
flags.DEFINE_boolean("include_rescaling", True, "Whether to include rescaling or not at the start of the model")
flags.DEFINE_string(
    "model_kwargs",
    "{}",
    "Keyword argument dictionary to pass to the constructor of the model being trained",
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

model = models.__dict__[FLAGS.model_name]
model = model(
        include_rescaling=FLAGS.include_rescaling,
        include_top=True,
        classes=1000,
        input_shape=(224, 224, 3),
        **eval(FLAGS.model_kwargs),
)

