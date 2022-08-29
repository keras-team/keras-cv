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
from keras_cv.models.object_detection.__internal__ import convert_inputs_to_tf_dataset
from keras_cv.models.object_detection.__internal__ import train_validation_split


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

    def fit(
        self,
        x=None,
        y=None,
        validation_data=None,
        validation_split=None,
        sample_weight=None,
        batch_size=None,
        **kwargs,
    ):
        dataset = convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
        )
        if validation_split and validation_data is None:
            (x, y, sample_weight,), validation_data = train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data is not None:
            val_x, val_y, val_sample = keras.utils.unpack_x_y_sample_weight(
                validation_data
            )
            validation_data = convert_inputs_to_tf_dataset(
                x=val_x, y=val_y, sample_weight=val_sample, batch_size=batch_size
            )

        dataset = dataset.map(self.encode_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return super().fit(x=dataset, validation_data=validation_data, **kwargs)

    def train_on_batch(self, x, y=None, **kwargs):
        x, y = self.encode_data(x, y)
        return super().train_on_batch(x=x, y=y, **kwargs)

    def test_on_batch(self, x, y=None, **kwargs):
        x, y = self.encode_data(x, y)
        return super().test_on_batch(x=x, y=y, **kwargs)

    def evaluate(
        self,
        x=None,
        y=None,
        sample_weight=None,
        batch_size=None,
        _use_cached_eval_dataset=None,
        **kwargs,
    ):
        dataset = convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
        )
        dataset = dataset.map(self.encode_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        # force _use_cached_eval_dataset=False
        # this is required to override evaluate().
        # We can remove _use_cached_eval_dataset=False when
        # https://github.com/keras-team/keras/issues/16958
        # is fixed
        return super().evaluate(x=dataset, _use_cached_eval_dataset=False, **kwargs)

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
