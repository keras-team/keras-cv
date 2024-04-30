# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities similar to tf.python.platform.resource_loader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import tensorflow as tf

TF_VERSION_FOR_ABI_COMPATIBILITY = "2.13"
abi_warning_already_raised = False


def get_project_root():
    """Returns project root folder."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_path_to_datafile(path):
    """Get the path to the specified file in the data dependencies.

    The path is relative to keras_cv/

    Args:
      path: a string resource path relative to keras_cv/
    Returns:
      The path to the specified data file
    """
    root_dir = get_project_root()
    return os.path.join(root_dir, path.replace("/", os.sep))


class LazySO:
    def __init__(self, relative_path):
        self.relative_path = relative_path
        self._ops = None

    @property
    def ops(self):
        if self._ops is None:
            self.display_warning_if_incompatible()
            self._ops = tf.load_op_library(
                get_path_to_datafile(self.relative_path)
            )
        return self._ops

    def display_warning_if_incompatible(self):
        global abi_warning_already_raised
        if abi_warning_already_raised or abi_is_compatible():
            return

        user_version = tf.__version__
        warnings.warn(
            f"You are currently using TensorFlow {user_version} and "
            f"trying to load a KerasCV custom op.\n"
            f"KerasCV has compiled its custom ops against TensorFlow "
            f"{TF_VERSION_FOR_ABI_COMPATIBILITY}, and there are no "
            f"compatibility guarantees between the two versions.\n"
            "This means that you might get segfaults when loading the custom "
            "op, or other kind of low-level errors.\n"
            "If you do, do not file an issue on Github. "
            "This is a known limitation.",
            UserWarning,
        )
        abi_warning_already_raised = True


def abi_is_compatible():
    return tf.__version__.startswith(TF_VERSION_FOR_ABI_COMPATIBILITY)
