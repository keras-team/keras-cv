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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.models.backbones.vit_det.vit_det_backbone import (
    ViTDetBackbone,
)
from keras_cv.src.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """VitDet{size}Backbone model.

    Reference:
        - [Detectron2](https://github.com/facebookresearch/detectron2)
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Example:
    ```python
    input_data = np.ones(shape=(1, 1024, 1024, 3))

    # Randomly initialized backbone
    model = VitDet{size}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.ViTDetBBackbone")
class ViTDetBBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("vitdet_base", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitdet_base_sa1b": copy.deepcopy(
                backbone_presets["vitdet_base_sa1b"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.ViTDetLBackbone")
class ViTDetLBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("vitdet_large", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitdet_large_sa1b": copy.deepcopy(
                backbone_presets["vitdet_large_sa1b"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.ViTDetHBackbone")
class ViTDetHBackbone(ViTDetBackbone):
    def __new__(
        cls,
        **kwargs,
    ):
        return ViTDetBackbone.from_preset("vitdet_huge", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitdet_huge_sa1b": copy.deepcopy(
                backbone_presets["vitdet_huge_sa1b"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(ViTDetBBackbone, "__doc__", ALIAS_DOCSTRING.format(size="B"))
setattr(ViTDetLBackbone, "__doc__", ALIAS_DOCSTRING.format(size="L"))
setattr(ViTDetHBackbone, "__doc__", ALIAS_DOCSTRING.format(size="H"))
