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
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_backbone import (
    MobileNetV3Backbone,
)
from keras_cv.src.models.backbones.mobilenet_v3.mobilenet_v3_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """MobileNetV3Backbone model with {num_layers} layers.

    References:
        - [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
        - [Based on the Original keras.applications MobileNetv3](https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet_v3.py)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer. Defaults to True.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = {name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.MobileNetV3SmallBackbone")
class MobileNetV3SmallBackbone(MobileNetV3Backbone):
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
        return MobileNetV3Backbone.from_preset("mobilenet_v3_small", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "mobilenet_v3_small_imagenet": copy.deepcopy(
                backbone_presets["mobilenet_v3_small_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return cls.presets


@keras_cv_export("keras_cv.models.MobileNetV3LargeBackbone")
class MobileNetV3LargeBackbone(MobileNetV3Backbone):
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
        return MobileNetV3Backbone.from_preset("mobilenet_v3_large", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "mobilenet_v3_large_imagenet": copy.deepcopy(
                backbone_presets["mobilenet_v3_large_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(
    MobileNetV3LargeBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MobileNetV3Large", num_layers="28"),
)
setattr(
    MobileNetV3SmallBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MobileNetV3Small", num_layers="14"),
)
