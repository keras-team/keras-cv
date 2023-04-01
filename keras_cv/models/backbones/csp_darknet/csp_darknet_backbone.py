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

"""DarkNet models for KerasCV.
Reference:
    - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
    - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
    - [YoloX Paper](https://arxiv.org/abs/2107.08430)
    - [YoloX implementation](https://github.com/ultralytics/yolov3)
"""
import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.__internal__.darknet_utils import CrossStagePartial
from keras_cv.models.__internal__.darknet_utils import DarknetConvBlock
from keras_cv.models.__internal__.darknet_utils import DarknetConvBlockDepthwise
from keras_cv.models.__internal__.darknet_utils import Focus
from keras_cv.models.__internal__.darknet_utils import (
    SpatialPyramidPoolingBottleneck,
)
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv.models")
class CSPDarkNetBackbone(Backbone):
    """Instantiates the CSPDarkNet architecture.

    Although the DarkNet architecture is commonly used for detection tasks, it
    is possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.
    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)
        - [YoloX implementation](https://github.com/ultralytics/yolov3)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning]
    (https://keras.io/guides/transfer_learning/).

    Args:
        depth_multiplier: A float value used to calculate the base depth of the
            model this changes based the detection model being used.
        width_multiplier: A float value used to calculate the base width of the
            model this changes based the detection model being used.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        use_depthwise: a boolean value used to decide whether a depthwise conv
            block should be used over a regular darknet block. Defaults to
            False.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple. Defaults to (None, None, 3).

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.CSPDarkNetBackbone.from_preset(
        "cspdarknettiny_imagenet"
    )
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = ResNetBackbone(
        depth_multiplier=0.33,
        width_multiplier=0.375,
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """

    def __init__(
        self,
        *,
        depth_multiplier,
        width_multiplier,
        include_rescaling,
        use_depthwise=False,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        ConvBlock = (
            DarknetConvBlockDepthwise if use_depthwise else DarknetConvBlock
        )

        base_channels = int(width_multiplier * 64)
        base_depth = max(round(depth_multiplier * 3), 1)

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
        # dark2
        x = ConvBlock(
            base_channels * 2, kernel_size=3, strides=2, name="dark2_conv"
        )(x)
        x = CrossStagePartial(
            base_channels * 2,
            num_bottlenecks=base_depth,
            use_depthwise=use_depthwise,
            name="dark2_csp",
        )(x)
        pyramid_level_inputs[2] = x

        # dark3
        x = ConvBlock(
            base_channels * 4, kernel_size=3, strides=2, name="dark3_conv"
        )(x)
        x = CrossStagePartial(
            base_channels * 4,
            num_bottlenecks=base_depth * 3,
            use_depthwise=use_depthwise,
            name="dark3_csp",
        )(x)
        pyramid_level_inputs[3] = x

        # dark4
        x = ConvBlock(
            base_channels * 8, kernel_size=3, strides=2, name="dark4_conv"
        )(x)
        x = CrossStagePartial(
            base_channels * 8,
            num_bottlenecks=base_depth * 3,
            use_depthwise=use_depthwise,
            name="dark4_csp",
        )(x)
        pyramid_level_inputs[4] = x

        # dark5
        x = ConvBlock(
            base_channels * 16, kernel_size=3, strides=2, name="dark5_conv"
        )(x)
        x = SpatialPyramidPoolingBottleneck(
            base_channels * 16,
            hidden_filters=base_channels * 8,
            name="dark5_spp",
        )(x)
        x = CrossStagePartial(
            base_channels * 16,
            num_bottlenecks=base_depth,
            residual=False,
            use_depthwise=use_depthwise,
            name="dark5_csp",
        )(x)
        pyramid_level_inputs[5] = x

        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs

        self.depth_multiplier = depth_multiplier
        self.width_multiplier = width_multiplier
        self.include_rescaling = include_rescaling
        self.use_depthwise = use_depthwise
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth_multiplier": self.depth_multiplier,
                "width_multiplier": self.width_multiplier,
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


ALIAS_DOCSTRING = """CSPDarkNetBackbone model with {depth_multiplier} depth
    multiplier and {width_multiplier} width multiplier.

    The CSPDarkNet architectures are commonly used for detection tasks. It is
    possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.
    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)
        - [YoloX implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning]
    (https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: bool, whether or not to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple. Defaults to (None, None, 3).

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = CSPDarkNetTinyBackbone()
    output = model(input_data)
    ```
"""


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
        return CSPDarkNetBackbone.from_preset("cspdarknettiny", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


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
        return CSPDarkNetBackbone.from_preset("cspdarknets", **kwargs)

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
        return CSPDarkNetBackbone.from_preset("cspdarknetm", **kwargs)

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
        return CSPDarkNetBackbone.from_preset("cspdarknetl", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class CSPDarkNetXBackbone(CSPDarkNetBackbone):
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
        return CSPDarkNetBackbone.from_preset("cspdarknetl", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(
    CSPDarkNetTinyBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(depth_multiplier="0.33", width_multiplier="0.375"),
)
setattr(
    CSPDarkNetSBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(depth_multiplier="0.33", width_multiplier="0.50"),
)
setattr(
    CSPDarkNetMBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(depth_multiplier="0.67", width_multiplier="0.75"),
)
setattr(
    CSPDarkNetLBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(depth_multiplier="1.00", width_multiplier="1.00"),
)
setattr(
    CSPDarkNetXBackbone,
    "__doc__",
    ALIAS_DOCSTRING.format(depth_multiplier="1.33", width_multiplier="1.25"),
)
