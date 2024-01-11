# Copyright 2023 The KerasCV Authors
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
import json
import os
import re

import tensorflow as tf
import tree
from absl.testing import parameterized

from keras_cv.backend import config
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.utils.tensor_utils import is_float_dtype
from keras_cv.utils.tensor_utils import standardize_dtype


def convert_to_comparible_type(x):
    """Convert tensors to comparable types.

    Any string are converted to plain python types. Any jax or torch tensors
    are converted to numpy.
    """
    if getattr(x, "dtype", None) == tf.string:
        if isinstance(x, tf.RaggedTensor):
            x = x.to_list()
        if isinstance(x, tf.Tensor):
            x = x.numpy() if x.shape.rank == 0 else x.numpy().tolist()
        return tree.map_structure(lambda x: x.decode("utf-8"), x)
    if isinstance(x, (tf.Tensor, tf.RaggedTensor)):
        return x
    if hasattr(x, "__array__"):
        return ops.convert_to_numpy(x)
    return x


def convert_to_numpy(x):
    if ops.is_tensor(x) and not isinstance(x, tf.RaggedTensor):
        return ops.convert_to_numpy(x)
    return x


class TestCase(tf.test.TestCase, parameterized.TestCase):
    """Base test case class for KerasCV. (Copied from KerasNLP)."""

    def assertAllGreaterEqual(self, x1, x2):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllGreaterEqual(x1, x2)

    def assertAllLessEqual(self, x1, x2):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllLessEqual(x1, x2)

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        # This metric dict hack is only needed for tf.keras, and can be
        # removed after we fully migrate to keras-core/Keras 3.
        if x1.__class__.__name__ == "_MetricDict":
            x1 = dict(x1)
        if x2.__class__.__name__ == "_MetricDict":
            x2 = dict(x2)
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)

    def assertEqual(self, x1, x2, msg=None):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertEqual(x1, x2, msg=msg)

    def assertAllEqual(self, x1, x2, msg=None):
        x1 = tree.map_structure(convert_to_comparible_type, x1)
        x2 = tree.map_structure(convert_to_comparible_type, x2)
        super().assertAllEqual(x1, x2, msg=msg)

    def assertDTypeEqual(self, x, expected_dtype, msg=None):
        input_dtype = standardize_dtype(x.dtype)
        super().assertEqual(input_dtype, expected_dtype, msg=msg)

    def run_layer_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        expected_output_data=None,
        expected_num_trainable_weights=0,
        expected_num_non_trainable_weights=0,
        expected_num_non_trainable_variables=0,
        run_training_check=True,
        run_mixed_precision_check=True,
    ):
        """Run basic tests for a modeling layer."""
        # Serialization test.
        layer = cls(**init_kwargs)
        self.run_serialization_test(layer)

        def run_build_asserts(layer):
            self.assertTrue(layer.built)
            self.assertLen(
                layer.trainable_weights,
                expected_num_trainable_weights,
                msg="Unexpected number of trainable_weights",
            )
            self.assertLen(
                layer.non_trainable_weights,
                expected_num_non_trainable_weights,
                msg="Unexpected number of non_trainable_weights",
            )
            self.assertLen(
                layer.non_trainable_variables,
                expected_num_non_trainable_variables,
                msg="Unexpected number of non_trainable_variables",
            )

        def run_output_asserts(layer, output, eager=False):
            output_shape = tree.map_structure(
                lambda x: None if x is None else x.shape, output
            )
            self.assertEqual(
                expected_output_shape,
                output_shape,
                msg="Unexpected output shape",
            )
            output_dtype = tree.flatten(output)[0].dtype
            self.assertEqual(
                standardize_dtype(layer.dtype),
                standardize_dtype(output_dtype),
                msg="Unexpected output dtype",
            )
            if eager and expected_output_data is not None:
                self.assertAllClose(expected_output_data, output)

        def run_training_step(layer, input_data, output_data):
            class TestModel(keras.Model):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

                def call(self, x):
                    if isinstance(x, dict):
                        return self.layer(**x)
                    else:
                        return self.layer(x)

            model = TestModel(layer)
            model.compile(optimizer="sgd", loss="mse", jit_compile=True)
            model.fit(input_data, output_data, verbose=0)

        if config.keras_3():
            # Build test.
            layer = cls(**init_kwargs)
            if isinstance(input_data, dict):
                shapes = {k + "_shape": v.shape for k, v in input_data.items()}
                layer.build(**shapes)
            else:
                layer.build(input_data.shape)
            run_build_asserts(layer)

            # Symbolic call test.
            keras_tensor_inputs = tree.map_structure(
                lambda x: keras.KerasTensor(x.shape, x.dtype), input_data
            )
            layer = cls(**init_kwargs)
            if isinstance(keras_tensor_inputs, dict):
                keras_tensor_outputs = layer(**keras_tensor_inputs)
            else:
                keras_tensor_outputs = layer(keras_tensor_inputs)
            run_build_asserts(layer)
            run_output_asserts(layer, keras_tensor_outputs)

        # Eager call test and compiled training test.
        layer = cls(**init_kwargs)
        if isinstance(input_data, dict):
            output_data = layer(**input_data)
        else:
            output_data = layer(input_data)
        run_output_asserts(layer, output_data, eager=True)

        if run_training_check:
            run_training_step(layer, input_data, output_data)

        # Never test mixed precision on torch CPU. Torch lacks support.
        if run_mixed_precision_check and config.backend() == "torch":
            import torch

            run_mixed_precision_check = torch.cuda.is_available()

        if run_mixed_precision_check:
            layer = cls(**{**init_kwargs, "dtype": "mixed_float16"})
            if isinstance(input_data, dict):
                output_data = layer(**input_data)
            else:
                output_data = layer(input_data)
            for tensor in tree.flatten(output_data):
                if is_float_dtype(tensor.dtype):
                    self.assertDTypeEqual(tensor, "float16")
            for weight in layer.weights:
                if is_float_dtype(weight.dtype):
                    self.assertDTypeEqual(weight, "float32")

    def run_preprocessing_layer_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output=None,
        expected_output_shape=None,
        batch_size=2,
    ):
        """Run basic tests for a preprocessing layer."""
        layer = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(layer)

        ds = tf.data.Dataset.from_tensor_slices(input_data)

        # Run with direct call.
        if isinstance(input_data, tuple):
            # Mimic tf.data unpacking behavior for preprocessing layers.
            output = layer(*input_data)
        else:
            output = layer(input_data)

        # Run with an unbatched dataset.
        output_ds = ds.map(layer).ragged_batch(1_000)
        self.assertAllClose(output, output_ds.get_single_element())

        # Run with a batched dataset.
        output_ds = ds.batch(1_000).map(layer)
        self.assertAllClose(output, output_ds.get_single_element())

        if expected_output is not None:
            self.assertAllClose(output, expected_output)

        if expected_output_shape is not None:
            self.assertAllClose(output.shape, expected_output_shape)

    def run_serialization_test(self, instance):
        """Check idempotency of serialize/deserialize.

        Not this is a much faster test than saving."""
        # get_config roundtrip
        cls = instance.__class__
        cfg = instance.get_config()
        cfg_json = json.dumps(cfg, sort_keys=True, indent=4)
        ref_dir = dir(instance)[:]
        revived_instance = cls.from_config(cfg)
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        # Dir tests only work on keras-3.
        if config.keras_3():
            self.assertEqual(ref_dir, dir(revived_instance))

        # serialization roundtrip
        serialized = keras.saving.serialize_keras_object(instance)
        serialized_json = json.dumps(serialized, sort_keys=True, indent=4)
        revived_instance = keras.saving.deserialize_keras_object(
            json.loads(serialized_json)
        )
        revived_cfg = revived_instance.get_config()
        revived_cfg_json = json.dumps(revived_cfg, sort_keys=True, indent=4)
        self.assertEqual(cfg_json, revived_cfg_json)
        # Dir tests only work on keras-3.
        if config.keras_3():
            new_dir = dir(revived_instance)[:]
            for lst in [ref_dir, new_dir]:
                if "__annotations__" in lst:
                    lst.remove("__annotations__")
            self.assertEqual(ref_dir, new_dir)

    def run_model_saving_test(
        self,
        cls,
        init_kwargs,
        input_data,
    ):
        """Save and load a model from disk and assert output is unchanged."""
        model = cls(**init_kwargs)
        model_output = model(input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, cls)

        # Check that output matches.
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)

    def run_task_test(
        self,
        cls,
        init_kwargs,
        train_data,
        expected_output_shape=None,
        batch_size=2,
    ):
        """Run basic tests for a backbone, including compilation."""
        task = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(task)
        ds = tf.data.Dataset.from_tensor_slices(train_data).batch(batch_size)
        x, y, sw = keras.utils.unpack_x_y_sample_weight(train_data)

        # Test predict.
        output = task.predict(x)
        if expected_output_shape is not None:
            output_shape = tree.map_structure(lambda x: x.shape, output)
            self.assertAllClose(output_shape, expected_output_shape)
        # With a dataset.
        output_ds = task.predict(ds)
        self.assertAllClose(output, output_ds)
        # Test fit.
        task.fit(x, y, sample_weight=sw)
        # With a dataset.
        task.fit(ds)

    def run_preset_test(
        self,
        cls,
        preset,
        input_data,
        init_kwargs={},
        expected_output=None,
        expected_output_shape=None,
    ):
        """Run instantiation and a forward pass for a preset."""
        self.assertRegex(cls.from_preset.__doc__, preset)

        with self.assertRaises(ValueError):
            cls.from_preset("clowntown", **init_kwargs)

        instance = cls.from_preset(preset, **init_kwargs)

        if isinstance(input_data, tuple):
            # Mimic tf.data unpacking behavior for preprocessing layers.
            output = instance(*input_data)
        else:
            output = instance(input_data)

        if isinstance(instance, keras.Model):
            instance = cls.from_preset(
                preset, load_weights=False, **init_kwargs
            )
            instance(input_data)

        if expected_output is not None:
            self.assertAllClose(output, expected_output)

        if expected_output_shape is not None:
            output_shape = tree.map_structure(lambda x: x.shape, output)
            self.assertAllClose(output_shape, expected_output_shape)

    def run_backbone_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output_shape,
        variable_length_data=None,
    ):
        """Run basic tests for a backbone, including compilation."""
        backbone = cls(**init_kwargs)
        # Check serialization (without a full save).
        self.run_serialization_test(backbone)
        # Call model eagerly.
        output = backbone(input_data)
        if isinstance(expected_output_shape, dict):
            for key in expected_output_shape:
                self.assertEqual(output[key].shape, expected_output_shape[key])
        else:
            self.assertEqual(output.shape, expected_output_shape)
        # Check compiled predict function.
        backbone.predict(input_data)
        # Convert to numpy first, torch GPU tensor -> tf.data will error.
        numpy_data = tree.map_structure(ops.convert_to_numpy, input_data)
        # Create a dataset.
        input_dataset = tf.data.Dataset.from_tensor_slices(numpy_data).batch(2)
        backbone.predict(input_dataset)

        # Check name maps to classname.
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", cls.__name__)
        name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
        self.assertRegexpMatches(backbone.name, name)

        # Test valid call with rescaling
        backbone = cls(**{**init_kwargs, "include_rescaling": True})
        backbone(self.input_data)

        # Test variable input channels
        for num_channels in [1, 4]:
            backbone = cls(**{**init_kwargs, "input_shape": num_channels})
            backbone(self.input_data)
