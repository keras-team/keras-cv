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
from keras_cv.src.models.backbones.efficientnet_lite.efficientnet_lite_backbone import (  # noqa: E501
    EfficientNetLiteBackbone,
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
    Example:
    ```python
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = {name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.EfficientNetLiteB0Backbone")
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
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b0", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetLiteB1Backbone")
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
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b1", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetLiteB2Backbone")
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
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b2", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetLiteB3Backbone")
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
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b3", **kwargs
        )

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.EfficientNetLiteB4Backbone")
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
        return EfficientNetLiteBackbone.from_preset(
            "efficientnetlite_b4", **kwargs
        )

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
    EfficientNetLiteB0Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetLiteB0"),
)
setattr(
    EfficientNetLiteB1Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetLiteB1"),
)
setattr(
    EfficientNetLiteB2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetLiteB2"),
)
setattr(
    EfficientNetLiteB3Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetLiteB3"),
)
setattr(
    EfficientNetLiteB4Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="EfficientNetLiteB4"),
)
