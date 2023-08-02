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

import copy

from keras_cv.models.backbones.detectron2.detectron2_backbone import (
    ViTDetBackbone,
)
from keras_cv.models.backbones.detectron2.detectron2_backbone_presets import (
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty

ALIAS_DOCSTRING = """{SAM}VitDet{size}Backbone model.

    Reference:
        - [Detectron2](https://github.com/facebookresearch/detectron2)
        - [Segment Anything](https://arxiv.org/abs/2304.02643)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Examples:
    ```python
    input_data = np.ones(shape=(1, 1024, 1024, 3))

    # Randomly initialized backbone
    model = {SAM}VitDet{size}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


class SAMViTDetBBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("sam_vitdet_b", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "sam_vitdet_b": copy.deepcopy(backbone_presets["sam_vitdet_b"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


class SAMViTDetLBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("sam_vitdet_l", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "sam_vitdet_l": copy.deepcopy(backbone_presets["sam_vitdet_l"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


class SAMViTDetHBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("sam_vitdet_h", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "sam_vitdet_h": copy.deepcopy(backbone_presets["sam_vitdet_h"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(
    SAMViTDetBBackbone, "__doc__", ALIAS_DOCSTRING.format(SAM="SAM", size="B")
)
setattr(
    SAMViTDetLBackbone, "__doc__", ALIAS_DOCSTRING.format(SAM="SAM", size="L")
)
setattr(
    SAMViTDetHBackbone, "__doc__", ALIAS_DOCSTRING.format(SAM="SAM", size="H")
)
