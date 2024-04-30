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
"""Benchmarks for training KerasCV models against the MNIST dataset."""

import time

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from keras_cv.src import models
from keras_cv.src.models.classification import image_classifier

# isort: off
from tensorflow.python.platform.benchmark import (
    ParameterizedBenchmark,
)


class ClassificationTrainingBenchmark(
    tf.test.Benchmark, metaclass=ParameterizedBenchmark
):
    """Benchmarks for classification models using `tf.test.Benchmark`."""

    _benchmark_parameters = [
        ("ResNet18V2Backbone", models.ResNet18V2Backbone),
    ]

    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.batch_size = 64
        # x shape is (batch_size, 56, 56, 3)
        # y shape is (batch_size, 10)
        self.dataset = (
            tfds.load("mnist", split="test")
            .map(
                lambda x: (
                    tf.image.grayscale_to_rgb(
                        tf.image.resize(x["image"], (56, 56))
                    ),
                    tf.one_hot(x["label"], self.num_classes),
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .batch(self.batch_size)
        )
        self.epochs = 1

    def benchmark_classification_training_single_gpu(self, app):
        self._run_benchmark(app, tf.distribute.OneDeviceStrategy("/gpu:0"))

    def benchmark_classification_training_multi_gpu(self, app):
        self._run_benchmark(app, tf.distribute.MirroredStrategy())

    def _run_benchmark(self, app, strategy):
        with strategy.scope():
            t0 = time.time()

            model = image_classifier.ImageClassifier(
                backbone=app(),
                num_classes=self.num_classes,
            )
            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            compile_time = time.time() - t0

        train_start_time = time.time()
        training_results = model.fit(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        train_end_time = time.time()

        training_time = train_end_time - train_start_time
        total_time = train_end_time - t0

        metrics = []
        metrics.append({"name": "compile_time", "value": compile_time})
        metrics.append(
            {"name": "avg_epoch_time", "value": training_time / self.epochs}
        )
        metrics.append({"name": "epochs", "value": self.epochs})
        metrics.append(
            {
                "name": "accuracy",
                "value": training_results.history["accuracy"][0],
            }
        )

        self.report_benchmark(wall_time=total_time, metrics=metrics)


if __name__ == "__main__":
    tf.test.main()
