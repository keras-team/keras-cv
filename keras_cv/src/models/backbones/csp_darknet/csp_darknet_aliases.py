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
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_backbone import (
    CSPDarkNetBackbone,
)
from keras_cv.src.models.backbones.csp_darknet.csp_darknet_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """CSPDarkNetBackbone model with {stackwise_channels} channels
    and {stackwise_depth} depths.

    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether or not to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, defaults to (None, None, 3).

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = CSPDarkNet{name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.CSPDarkNetTinyBackbone")
class CSPDarkNetTinyBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("csp_darknet_tiny", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "csp_darknet_tiny_imagenet": copy.deepcopy(
                backbone_presets["csp_darknet_tiny_imagenet"]
            )
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.CSPDarkNetSBackbone")
class CSPDarkNetSBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("csp_darknet_s", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.CSPDarkNetMBackbone")
class CSPDarkNetMBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("csp_darknet_m", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


@keras_cv_export("keras_cv.models.CSPDarkNetLBackbone")
class CSPDarkNetLBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("csp_darknet_l", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "csp_darknet_l_imagenet": copy.deepcopy(
                backbone_presets["csp_darknet_l_imagenet"]
            )
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.CSPDarkNetXLBackbone")
class CSPDarkNetXLBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("csp_darknet_xl", **kwargs)

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
    CSPDarkNetTinyBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="Tiny",
        stackwise_channels="[48, 96, 192, 384]",
        stackwise_depth="[1, 3, 3, 1]",
    ),
)
setattr(
    CSPDarkNetSBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="S",
        stackwise_channels="[64, 128, 256, 512]",
        stackwise_depth="[1, 3, 3, 1]",
    ),
)
setattr(
    CSPDarkNetMBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="M",
        stackwise_channels="[96, 192, 384, 768]",
        stackwise_depth="[2, 6, 6, 2]",
    ),
)
setattr(
    CSPDarkNetLBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="L",
        stackwise_channels="[128, 256, 512, 1024]",
        stackwise_depth="[3, 9, 9, 3]",
    ),
)
setattr(
    CSPDarkNetXLBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(
        name="XL",
        stackwise_channels="[170, 340, 680, 1360]",
        stackwise_depth="[4, 12, 12, 4]",
    ),
)
