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

import tensorflow as tf
import tree
from absl.testing import parameterized

from keras_cv.backend import config
from keras_cv.backend import keras
from keras_cv.backend import ops
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

    def run_preprocessing_layer_test(
        self,
        cls,
        init_kwargs,
        input_data,
        expected_output=None,
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

        if expected_output:
            self.assertAllClose(output, expected_output)

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
        # Dir tests only work on keras-core.
        if config.multi_backend():
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
        # Dir tests only work on keras-core.
        if config.multi_backend():
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


def convert_to_numpy(x):
    if ops.is_tensor(x) and not isinstance(x, tf.RaggedTensor):
        return ops.convert_to_numpy(x)
    return x
