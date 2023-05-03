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

from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone import (
    EfficientNetV2Backbone,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.models.backbones.efficientnet_v2.efficientnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty

ALIAS_BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298)
      (ICML 2021)

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
    Returns:
      A `keras.Model` instance.
"""  # noqa: E501


class EfficientNetV2SBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-s", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-s": copy.deepcopy(
                backbone_presets["efficientnetv2-s"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-s_imagenet": copy.deepcopy(
                backbone_presets_with_weights["efficientnetv2-s_imagenet"]
            ),
        }


class EfficientNetV2MBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-m", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-m": copy.deepcopy(
                backbone_presets["efficientnetv2-m"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2LBackbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-l", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-l": copy.deepcopy(
                backbone_presets["efficientnetv2-l"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetV2B0Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-b0": copy.deepcopy(
                backbone_presets["efficientnetv2-b0"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b0_imagenet": copy.deepcopy(
                backbone_presets_with_weights["efficientnetv2-b0_imagenet"]
            ),
        }


class EfficientNetV2B1Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-b1": copy.deepcopy(
                backbone_presets["efficientnetv2-b1"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b1_imagenet": copy.deepcopy(
                backbone_presets_with_weights["efficientnetv2-b1_imagenet"]
            ),
        }


class EfficientNetV2B2Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetv2-b2": copy.deepcopy(
                backbone_presets["efficientnetv2-b2"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {
            "efficientnetv2-b2_imagenet": copy.deepcopy(
                backbone_presets_with_weights["efficientnetv2-b2_imagenet"]
            ),
        }


class EfficientNetV2B3Backbone(EfficientNetV2Backbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return EfficientNetV2Backbone.from_preset("efficientnetv2-b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


EfficientNetV2B0Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2B0"
)
EfficientNetV2B1Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2B1"
)
EfficientNetV2B2Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2B2"
)
EfficientNetV2B3Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2B3"
)
EfficientNetV2SBackbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2S"
)
EfficientNetV2MBackbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2M"
)
EfficientNetV2LBackbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetV2L"
)
