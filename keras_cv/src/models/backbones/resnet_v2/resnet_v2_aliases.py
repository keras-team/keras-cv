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
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_backbone import (
    ResNetV2Backbone,
)
from keras_cv.src.models.backbones.resnet_v2.resnet_v2_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """ResNetV2Backbone model with {num_layers} layers.

    Reference:
        - [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016)

    The difference in ResNet and ResNetV2 rests in the structure of their
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
    model = ResNet{num_layers}V2Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.ResNet18V2Backbone")
class ResNet18V2Backbone(ResNetV2Backbone):
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
        return ResNetV2Backbone.from_preset("resnet18_v2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet34V2Backbone")
class ResNet34V2Backbone(ResNetV2Backbone):
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
        return ResNetV2Backbone.from_preset("resnet34_v2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet50V2Backbone")
class ResNet50V2Backbone(ResNetV2Backbone):
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
        return ResNetV2Backbone.from_preset("resnet50_v2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "resnet50_v2_imagenet": copy.deepcopy(
                backbone_presets["resnet50_v2_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.ResNet101V2Backbone")
class ResNet101V2Backbone(ResNetV2Backbone):
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
        return ResNetV2Backbone.from_preset("resnet101_v2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.ResNet152V2Backbone")
class ResNet152V2Backbone(ResNetV2Backbone):
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
        return ResNetV2Backbone.from_preset("resnet152_v2", **kwargs)

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
    ResNet18V2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers=18),
)
setattr(
    ResNet34V2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers=34),
)
setattr(
    ResNet50V2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers=50),
)
setattr(
    ResNet101V2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers=101),
)
setattr(
    ResNet152V2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(num_layers=152),
)
