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

import keras_core
import pytest
import tensorflow as tf
from packaging import version

from keras_cv.backend import config as backend_config
from keras_cv.backend.config import keras_3


def pytest_addoption(parser):
    parser.addoption(
        "--run_large",
        action="store_true",
        default=False,
        help="run large tests",
    )
    parser.addoption(
        "--run_extra_large",
        action="store_true",
        default=False,
        help="run extra_large tests",
    )
    parser.addoption(
        "--check_gpu",
        action="store_true",
        default=False,
        help="fail if a gpu is not present",
    )


def pytest_configure(config):
    # Verify that device has GPU and detected by backend
    if config.getoption("--check_gpu"):
        found_gpu = False
        backend = backend_config.backend()
        if backend == "jax":
            import jax

            try:
                found_gpu = bool(jax.devices("gpu"))
            except RuntimeError:
                found_gpu = False
        elif backend == "tensorflow":
            found_gpu = bool(tf.config.list_logical_devices("GPU"))
        elif backend == "torch":
            import torch

            found_gpu = bool(torch.cuda.device_count())
        if not found_gpu:
            pytest.fail(f"No GPUs discovered on the {backend} backend.")

    config.addinivalue_line(
        "markers", "large: mark test as being slow or requiring a network"
    )
    config.addinivalue_line(
        "markers",
        "extra_large: mark test as being too large to run continuously",
    )
    config.addinivalue_line(
        "markers",
        "tf_keras_only: mark test as a Keras 2-only test",
    )
    config.addinivalue_line(
        "markers",
        "tf_only: mark test as a Tensorflow-only test",
    )


def pytest_collection_modifyitems(config, items):
    run_extra_large_tests = config.getoption("--run_extra_large")
    # Run large tests for --run_extra_large or --run_large.
    run_large_tests = config.getoption("--run_large") or run_extra_large_tests

    # Run Keras saving tests on 2.12 stable, nightlies and later releases.
    skip_keras_saving_test = pytest.mark.skipif(
        version.parse(tf.__version__) < version.parse("2.12.0-dev0"),
        reason="keras_v3 format requires tf > 2.12.",
    )
    skip_large = pytest.mark.skipif(
        not run_large_tests, reason="need --run_large option to run"
    )
    skip_extra_large = pytest.mark.skipif(
        not run_extra_large_tests, reason="need --run_extra_large option to run"
    )
    skip_keras_2_only = pytest.mark.skipif(
        keras_3(),
        reason="This test is only supported on Keras 2",
    )
    skip_tf_only = pytest.mark.skipif(
        keras_3() and keras_core.backend.backend() != "tensorflow",
        reason="This test is only supported on TensorFlow",
    )
    for item in items:
        if "keras_format" in item.name:
            item.add_marker(skip_keras_saving_test)
        if "tf_format" in item.name:
            item.add_marker(skip_extra_large)
        if "large" in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
        if "tf_keras_only" in item.keywords:
            item.add_marker(skip_keras_2_only)
        if "tf_only" in item.keywords:
            item.add_marker(skip_tf_only)
