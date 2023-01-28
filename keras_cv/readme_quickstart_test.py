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
import tensorflow_datasets as tfds
from tensorflow import keras

import keras_cv

"""
ATTENTION!!!

This test exists to make sure we don't accidentally break our quickstart pages.

If you change this test you are responsible for creating a PR to update:
- https://github.com/keras-team/keras-io/blob/master/templates/keras_cv/index.md
- https://github.com/keras-team/keras-cv/blob/master/README.md#quickstart

Thank you!
"""


def test_quickstart_runs():
    augmenter = keras_cv.layers.Augmenter(
        layers=[
            keras_cv.layers.RandomFlip(),
            keras_cv.layers.RandAugment(value_range=(0, 255)),
            keras_cv.layers.CutMix(),
            keras_cv.layers.MixUp(),
        ]
    )

    def augment_data(images, labels):
        labels = tf.one_hot(labels, 3)
        inputs = {"images": images, "labels": labels}
        outputs = augmenter(inputs)
        return outputs["images"], outputs["labels"]

    dataset = tfds.load("rock_paper_scissors", as_supervised=True, split="train")
    dataset = dataset.batch(64)
    dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    densenet = keras_cv.models.DenseNet121(
        include_rescaling=True, include_top=True, classes=3
    )
    densenet.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # In the real example remove the take(1) call below:
    densenet.fit(dataset.take(1))
