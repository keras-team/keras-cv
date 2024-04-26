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

try:
    from keras.src.utils import tf_utils
except ImportError:
    from keras.utils import tf_utils


def _minimum_control_deps(outputs):
    """Returns the minimum control dependencies to ensure step succeeded."""
    if tf.executing_eagerly():
        return []  # Control dependencies not needed.
    outputs = tf.nest.flatten(outputs, expand_composites=True)
    for out in outputs:
        # Variables can't be control dependencies.
        if not isinstance(out, tf.Variable):
            return [out]  # Return first Tensor or Op from outputs.
    return []  # No viable Tensor or Op to use for control deps.


def make_predict_function(model, force=False):
    if model.predict_function is not None and not force:
        return model.predict_function

    def step_function(iterator):
        """Runs a single evaluation step."""

        def run_step(data):
            outputs = model.predict_step(data)
            # Ensure counter is updated only if `test_step` succeeds.
            with tf.control_dependencies(_minimum_control_deps(outputs)):
                model._predict_counter.assign_add(1)
            return outputs

        if model._jit_compile:
            run_step = tf.function(
                run_step, jit_compile=True, reduce_retracing=True
            )

        data = next(iterator)
        outputs = model.distribute_strategy.run(run_step, args=(data,))
        outputs = model.distribute_strategy.gather(outputs, axis=0)
        # Note that this is the only deviation from the base keras.Model
        # implementation. We add the decode_step inside of the computation
        # graph but outside of the distribute_strategy (i.e on host CPU).
        if not isinstance(data, tf.Tensor):
            data = tf.concat(data.values, axis=0)
        return model.decode_predictions(outputs, data)

    # Special case if steps_per_execution is one.
    if (
        model._steps_per_execution is None
        or model._steps_per_execution.numpy().item() == 1
    ):

        def predict_function(iterator):
            """Runs an evaluation execution with a single step."""
            return step_function(iterator)

    else:

        def predict_function(iterator):
            """Runs an evaluation execution with multiple steps."""
            outputs = step_function(iterator)
            for _ in tf.range(model._steps_per_execution - 1):
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
                step_outputs = step_function(iterator)
                outputs = tf.nest.map_structure(
                    lambda t1, t2: tf.concat([t1, t2]), outputs, step_outputs
                )
            return outputs

    if not model.run_eagerly:
        predict_function = tf.function(predict_function, reduce_retracing=True)
    model.predict_function = predict_function

    return predict_function
