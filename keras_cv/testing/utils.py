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
import os


def integration(test_fn):
    # local scoped import to make installation only required in testing mode
    import pytest

    return pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Integration test.  Set environment variable `INTEGRATION=True`"
        " to run.",
    )(test_fn)


def requires_custom_ops(test_fn):
    import pytest

    return pytest.mark.skipif(
        "SKIP_CUSTOM_OPS" not in os.environ or os.environ["SKIP_CUSTOM_OPS"] != "true",
        reason="Requires custom ops.  Set environment variable `SKIP_CUSTOM_OPS=True`"
        " to run.",
    )(test_fn)
