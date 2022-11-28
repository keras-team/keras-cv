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
from dataclasses import dataclass

v1 = {
    "text-encoder": "",
    "diffusion-model": "",
    "image-encoder": "",
    "image-decoder": "",
}

v1point5 = {
    "text-encoder": "",
    "diffusion-model": "",
    "image-encoder": "",
    "image-decoder": "",
}

all_weights = {"v1": v1, "v1.5": v1point5}


def parse_weights(weights):
    if weights is None:
        return weights
    if isinstance(weights, StableDiffusionWeights):
        return weights

    if not weights in all_weights:
        raise ValueError(
            f"Expected `weights` to be one of {list(all_weights.keys())}. "
            f"Got `weights={weights}`."
        )

    return all_weights[weights]
