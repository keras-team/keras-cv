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

import pytest

from keras_cv import version_check


def test_check_tf_version_error():
    version_check.tf.__version__ = "2.8.0"

    with pytest.warns(ImportWarning) as record:
        version_check.check_tf_version()
    assert len(record) == 1
    assert (
        "Tensorflow package version needs to be at least 2.9.0"
        in record[0].message.args[0]
    )


def test_check_tf_version_passes_rc2():
    # should pass
    version_check.tf.__version__ = "2.9.0rc2"
    version_check.check_tf_version()


def test_check_tf_version_passes_nightly():
    # should pass
    version_check.tf.__version__ = "2.10.0-dev20220419"
    version_check.check_tf_version()
