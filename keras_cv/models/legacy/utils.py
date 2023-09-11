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
"""Utility functions for models"""

from tensorflow import keras
from tensorflow.keras import layers


def parse_model_inputs(input_shape, input_tensor):
    if input_tensor is None:
        return layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            return layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            return input_tensor


def as_backbone(self, min_level=None, max_level=None):
    """Convert the application model into a model backbone for other tasks.
    The backbone model will usually take same inputs as the original application
    model, but produce multiple outputs, one for each feature level. Those
    outputs can be feed to network downstream, like FPN and RPN. The output of
    the backbone model will be a dict with int as key and tensor as value. The
    int key represent the level of the feature output. A typical feature pyramid
    has five levels corresponding to scales P3, P4, P5, P6, P7 in the backbone.
    Scale Pn represents a feature map 2n times smaller in width and height than
    the input image.
    Args:
        min_level: optional int, the lowest level of feature to be included in
            the output, defaults to model's lowest feature level
            (based on the model structure).
        max_level: optional int, the highest level of feature to be included in
            the output, defaults to model's highest feature level
            (based on the model structure).
    Returns:
        a `keras.Model` which has dict as outputs.
    Raises:
        ValueError: When the model is lack of information for feature level, and
        can't be converted to backbone model, or the min_level/max_level param
        is out of range based on the model structure.
    """
    if hasattr(self, "_backbone_level_outputs"):
        backbone_level_outputs = self._backbone_level_outputs
        model_levels = list(sorted(backbone_level_outputs.keys()))
        if min_level is not None:
            if min_level < model_levels[0]:
                raise ValueError(
                    f"The min_level provided: {min_level} should be in "
                    f"the range of {model_levels}"
                )
        else:
            min_level = model_levels[0]

        if max_level is not None:
            if max_level > model_levels[-1]:
                raise ValueError(
                    f"The max_level provided: {max_level} should be in "
                    f"the range of {model_levels}"
                )
        else:
            max_level = model_levels[-1]

        outputs = {}
        for level in range(min_level, max_level + 1):
            outputs[level] = backbone_level_outputs[level]

        return keras.Model(inputs=self.inputs, outputs=outputs)

    else:
        raise ValueError(
            "The current model doesn't have any feature level "
            "information and can't be convert to backbone model."
        )
