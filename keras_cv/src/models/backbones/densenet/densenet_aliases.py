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
from keras_cv.src.models.backbones.densenet.densenet_backbone import (
    DenseNetBackbone,
)
from keras_cv.src.models.backbones.densenet.densenet_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """DenseNetBackbone model with {num_layers} layers.

    Reference:
        - [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

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
    model = DenseNet{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.DenseNet121Backbone")
class DenseNet121Backbone(DenseNetBackbone):
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
        return DenseNetBackbone.from_preset("densenet121", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "densenet121_imagenet": copy.deepcopy(
                backbone_presets["densenet121_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return cls.presets


@keras_cv_export("keras_cv.models.DenseNet169Backbone")
class DenseNet169Backbone(DenseNetBackbone):
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
        return DenseNetBackbone.from_preset("densenet169", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "densenet169_imagenet": copy.deepcopy(
                backbone_presets["densenet169_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return cls.presets


@keras_cv_export("keras_cv.models.DenseNet201Backbone")
class DenseNet201Backbone(DenseNetBackbone):
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
        return DenseNetBackbone.from_preset("densenet201", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "densenet201_imagenet": copy.deepcopy(
                backbone_presets["densenet201_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return cls.presets


setattr(DenseNet121Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=121))
setattr(DenseNet169Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=169))
setattr(DenseNet201Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=201))
