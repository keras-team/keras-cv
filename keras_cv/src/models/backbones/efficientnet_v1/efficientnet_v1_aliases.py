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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.models.backbones.efficientnet_v1.efficientnet_v1_backbone import (  # noqa: E501
    EfficientNetV1Backbone,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """Instantiates the {name} architecture.

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


@keras_cv_export("keras_cv.models.EfficientNetV1B0Backbone")
class EfficientNetV1B0Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B1Backbone")
class EfficientNetV1B1Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B2Backbone")
class EfficientNetV1B2Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B3Backbone")
class EfficientNetV1B3Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B4Backbone")
class EfficientNetV1B4Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b4", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B5Backbone")
class EfficientNetV1B5Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b5", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B6Backbone")
class EfficientNetV1B6Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b6", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetV1B7Backbone")
class EfficientNetV1B7Backbone(EfficientNetV1Backbone):
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
        return EfficientNetV1Backbone.from_preset("efficientnetv1_b7", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


setattr(
    EfficientNetV1B0Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B0"),
)
setattr(
    EfficientNetV1B1Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B1"),
)
setattr(
    EfficientNetV1B2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B2"),
)
setattr(
    EfficientNetV1B3Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B3"),
)
setattr(
    EfficientNetV1B4Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B4"),
)
setattr(
    EfficientNetV1B5Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B5"),
)
setattr(
    EfficientNetV1B6Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B6"),
)
setattr(
    EfficientNetV1B7Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetV1B7"),
)
