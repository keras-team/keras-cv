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

"""CSPDarkNet models for KerasCV. """
import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    CrossStagePartial,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlock,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlockDepthwise,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import Focus
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    SpatialPyramidPoolingBottleneck,
)
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv.models")
class CSPDarkNetBackbone(Backbone):
    """This class represents the CSPDarkNet architecture.

    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/abs/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_channels: A list of ints, the number of channels for each dark
            level in the model.
        stackwise_depth: A list of ints, the depth for each dark level in the
            model.
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        use_depthwise: bool, whether a `DarknetConvBlockDepthwise` should be
            used over a `DarknetConvBlock`, defaults to False.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Returns:
        A `keras.Model` instance.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.CSPDarkNetBackbone.from_preset(
        "csp_darknet_tiny_imagenet"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = keras_cv.models.CSPDarkNetBackbone(
        stackwise_channels=[128, 256, 512, 1024],
        stackwise_depth=[3, 9, 9, 3],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        stackwise_channels,
        stackwise_depth,
        include_rescaling,
        use_depthwise=False,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        base_channels = stackwise_channels[0] // 2

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        # stem
        x = Focus(name="stem_focus")(x)
        x = DarknetConvBlock(
            base_channels, kernel_size=3, strides=1, name="stem_conv"
        )(x)

        pyramid_level_inputs = {}
        for index, (channels, depth) in enumerate(
            zip(stackwise_channels, stackwise_depth)
        ):
            x = ConvBlock(
                channels,
                kernel_size=3,
                strides=2,
                name=f"dark{index + 2}_conv",
            )(x)

            if index == len(stackwise_depth) - 1:
                x = SpatialPyramidPoolingBottleneck(
                    channels,
                    hidden_filters=channels // 2,
                    name=f"dark{index + 2}_spp",
                )(x)

            x = CrossStagePartial(
                channels,
                num_bottlenecks=depth,
                use_depthwise=use_depthwise,
                residual=(index != len(stackwise_depth) - 1),
                name=f"dark{index + 2}_csp",
            )(x)
            pyramid_level_inputs[index + 2] = x.node.layer.name

        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs

        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.include_rescaling = include_rescaling
        self.use_depthwise = use_depthwise
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_channels": self.stackwise_channels,
                "stackwise_depth": self.stackwise_depth,
                "include_rescaling": self.include_rescaling,
                "use_depthwise": self.use_depthwise,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)


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

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = CSPDarkNet{name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


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
