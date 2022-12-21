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


class ObjectDetectionBaseModel(keras.Model):
    """ObjectDetectionBaseModel performs asynchonous label encoding.

    ObjectDetectionBaseModel invokes the provided `label_encoder` in the `tf.data`
    pipeline to ensure optimal training performance.  This is done by overriding the
    methods `train_on_batch()`, `fit()`, `test_on_batch()`, and `evaluate()`.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
