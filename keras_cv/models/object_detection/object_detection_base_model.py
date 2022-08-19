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
from tensorflow import keras

from keras_cv import bounding_box


class ObjectDetectionBaseModel(keras.Model):

    """ObjectDetectionBaseModel performs asynchonous label encoding.

    ObjectDetectionBaseModel invokes the provided `label_encoder` in the `tf.data`
    pipeline to ensure optimal training performance.  This is done by overriding the
    methods `train_on_batch()`, `fit()`, `test_on_batch()`, and `evaluate()`.

    """

    def __init__(self, bounding_box_format, label_encoder, **kwargs):
        super().__init__(**kwargs)
        self.bounding_box_format = bounding_box_format
        self.label_encoder = label_encoder

        self.label_encoder.build((None, None, None))

    def fit(self, x=None, y=None, validation_data=None, sample_weight=None, batch_size=None, **kwargs):
        dataset = _convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
        )

        if validation_data is not None:
            validation_data = _convert_inputs_to_tf_dataset(*validation_data, batch_size=batch_size)
            validation_data = validation_data.map(self.encode_data, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(self.encode_data, num_parallel_calls=tf.data.AUTOTUNE)
        return super().fit(x=dataset, **kwargs)

    def evaluate(self, x=None, y=None, sample_weight=None, batch_size=None, **kwargs):
        dataset = _convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
        )
        dataset = dataset.map(self.encode_data, num_parallel_calls=tf.data.AUTOTUNE)
        return super().evaluate(x=dataset, **kwargs)

    def encode_data(self, x, y):
        y_for_metrics = y

        y = bounding_box.convert_format(
            y,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        y_training_target = self.label_encoder(x, y)
        y_training_target = bounding_box.convert_format(
            y_training_target,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )
        return x, (y_for_metrics, y_training_target)


def _convert_inputs_to_tf_dataset(x=None, y=None, sample_weight=None, batch_size=None):
    if sample_weight is not None:
        raise ValueError("RetinaNet does not yet support `sample_weight`.")

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not provide a value for "
                f"`y` or `batch_size`.  Got `y={y}`, `batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset
