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


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def get_plt():
    """returns matplotlib, or raises an ImportError if matplotlib is not found."""
    if plt is None:
        raise ImportError(
            "matplotlib is not installed in the "
            "environment. Please use `pip install matplotlib` to install matplotlib "
            "if you would like to use KerasCV visualization tools"
        )
    return plt
