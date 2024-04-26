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
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone import (
    ResNetBackbone,
)
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """ResNetBackbone (V1) model with {num_layers} layers.

    Reference:
        - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    The difference in ResNetV1 and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNetV1 where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = ResNet{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.ResNet18Backbone")
class ResNet18Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet18", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet34Backbone")
class ResNet34Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet34", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet50Backbone")
class ResNet50Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet50", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "resnet50_imagenet": copy.deepcopy(
                backbone_presets["resnet50_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.ResNet101Backbone")
class ResNet101Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet101", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet152Backbone")
class ResNet152Backbone(ResNetBackbone):
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
        return ResNetBackbone.from_preset("resnet152", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


setattr(ResNet18Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=18))
setattr(ResNet34Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=34))
setattr(ResNet50Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=50))
setattr(ResNet101Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=101))
setattr(ResNet152Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=152))
