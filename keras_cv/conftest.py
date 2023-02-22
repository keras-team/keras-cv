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
import sys

import pytest
import tensorflow as tf
from packaging import version


@pytest.fixture(scope="class")
def tpu_test_class(request, tpu_strategy):
    # set a class attribute on the invoking test context
    request.cls.tpu_strategy = tpu_strategy


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "large: mark test as being slow or requiring a network"
    )
    config.addinivalue_line(
        "markers",
        "extra_large: mark test as being too large to run continuously",
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
    for item in items:
        if "keras_format" in item.name:
            item.add_marker(skip_keras_saving_test)
        if "large" in item.keywords:
            item.add_marker(skip_large)
        if "extra_large" in item.keywords:
            item.add_marker(skip_extra_large)
