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

from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_backbone import (
    EfficientNetLiteBackbone,
)
from keras_cv.models.backbones.efficientnet_lite.efficientnet_lite_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty

ALIAS_BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
    - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
      (ICML 2019)

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
"""  # noqa: E501


class EfficientNetLiteB0Backbone(EfficientNetLiteBackbone):
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
        return EfficientNetLiteBackbone.from_preset("efficientnetlite_b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetlite_b0": copy.deepcopy(
                backbone_presets["efficientnetlite_b0"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetLiteB1Backbone(EfficientNetLiteBackbone):
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
        return EfficientNetLiteBackbone.from_preset("efficientnetv1_b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetlite_b1": copy.deepcopy(
                backbone_presets["efficientnetlite_b1"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetLiteB2Backbone(EfficientNetLiteBackbone):
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
        return EfficientNetLiteBackbone.from_preset("efficientnetlite_b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetlite_b2": copy.deepcopy(
                backbone_presets["efficientnetlite_b2"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetLiteB3Backbone(EfficientNetLiteBackbone):
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
        return EfficientNetLiteBackbone.from_preset("efficientnetlite_b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetlite_b3": copy.deepcopy(
                backbone_presets["efficientnetlite_b3"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class EfficientNetLiteB4Backbone(EfficientNetLiteBackbone):
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
        return EfficientNetLiteBackbone.from_preset("efficientnetlite_b4", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "efficientnetlite_b4": copy.deepcopy(
                backbone_presets["efficientnetlite_b4"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}
    
EfficientNetLiteB0Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetLiteB0"
)
EfficientNetLiteB1Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetLiteB1"
)
EfficientNetLiteB2Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetLiteB2"
)
EfficientNetLiteB3Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetLiteB3"
)
EfficientNetLiteB4Backbone.__doc__ = ALIAS_BASE_DOCSTRING.format(
    name="EfficientNetLiteB4"
)