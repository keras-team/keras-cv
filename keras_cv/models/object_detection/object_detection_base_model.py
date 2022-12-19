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
from keras.engine.training import _minimum_control_deps
from keras.engine.training import reduce_per_replica
from keras.utils import tf_utils
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
        if validation_split and validation_data is None:
            (x, y, sample_weight,), validation_data = train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )
        dataset = convert_inputs_to_tf_dataset(
            x=x, y=y, sample_weight=sample_weight, batch_size=batch_size
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
        gt_boxes = y[..., :4]
        gt_classes = y[..., 4]
        gt_boxes = bounding_box.convert_format(
            gt_boxes,
            source=self.bounding_box_format,
            target=self.label_encoder.bounding_box_format,
            images=x,
        )
        box_targets, class_targets = self.label_encoder(x, gt_boxes, gt_classes)
        box_targets = bounding_box.convert_format(
            box_targets,
            source=self.label_encoder.bounding_box_format,
            target=self.bounding_box_format,
            images=x,
        )
        return x, {"boxes": box_targets, "classes": class_targets}

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def step_function(model, iterator):
            """Runs a single evaluation step."""

            def run_step(data):
                outputs = model.predict_step(data)
                # Ensure counter is updated only if `test_step` succeeds.
                with tf.control_dependencies(_minimum_control_deps(outputs)):
                    model._predict_counter.assign_add(1)
                return outputs

            if self._jit_compile:
                run_step = tf.function(
                    run_step, jit_compile=True, reduce_retracing=True
                )

            data = next(iterator)
            outputs = model.distribute_strategy.run(run_step, args=(data,))
            outputs = reduce_per_replica(
                outputs, self.distribute_strategy, reduction="concat"
            )
            # Note that this is the only deviation from the base keras.Model
            # implementation. We add the decode_step inside of the computation
            # graph but outside of the distribute_strategy (i.e on host CPU).
            if not isinstance(data, tf.Tensor):
                data = tf.concat(data.values, axis=0)
            return self.decode_predictions(outputs, data)

        # Special case if steps_per_execution is one.
        if (
            self._steps_per_execution is None
            or self._steps_per_execution.numpy().item() == 1
        ):

            def predict_function(iterator):
                """Runs an evaluation execution with a single step."""
                return step_function(self, iterator)

        else:

            def predict_function(iterator):
                """Runs an evaluation execution with multiple steps."""
                outputs = step_function(self, iterator)
                for _ in tf.range(self._steps_per_execution - 1):
                    tf.autograph.experimental.set_loop_options(
                        shape_invariants=[
                            (
                                outputs,
                                tf.nest.map_structure(
                                    lambda t: tf_utils.get_tensor_spec(
                                        t, dynamic_batch=True
                                    ).shape,
                                    outputs,
                                ),
                            )
                        ]
                    )
                    step_outputs = step_function(self, iterator)
                    outputs = tf.nest.map_structure(
                        lambda t1, t2: tf.concat([t1, t2]), outputs, step_outputs
                    )
                return outputs

        if not self.run_eagerly:
            predict_function = tf.function(predict_function, reduce_retracing=True)
        self.predict_function = predict_function

        return self.predict_function

    def decode_predictions(self, predictions, images):
        """Decode predictions (e.g. with an NmsPredictionDecoder).

        By default, this returns raw training predictions.
        Subclasses that which to return decoded boxes should override this method.
        """

        return predictions
