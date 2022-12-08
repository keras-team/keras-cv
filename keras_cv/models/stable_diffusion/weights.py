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
from tensorflow import keras


def load_weights(model, weights, preconfigured_weights, hashes):
    """internal utility to load weights to a StableDiffusion model.

    Args:
        model: the Keras model to load weights into
        preconfigured_weights: a dictionary of names -> http urls containing
            the weights
        hashes: hashes for the http urls containing the weights
    """
    if weights is None:
        return
    hash = None
    if weights in preconfigured_weights:
        weights = preconfigured_weights[weights]
        if weights not in hashes:
            raise ValueError(
                f"Expected a hash for weights `weights={weights}`. "
                "Users should never see this error, open a GitHub issue if you receive "
                "this error message."
            )
        hash = hashes[weights]

    # support for remote file loading
    if weights.startswith("http"):
        weights = keras.utils.get_file(
            origin=weights,
            file_hash=hash,
        )

    model.load_weights(weights)


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
