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
    - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
    - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
"""

import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    DarknetConvBlock,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    ResidualBlocks,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_utils import (
    SpatialPyramidPoolingBottleneck,
)
from keras_cv.models.backbones.darknet.darknet_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.darknet.darknet_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.legacy import utils
from keras_cv.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_cv.models")
class DarkNetBackbone(Backbone):

    """Represents the DarkNet architecture.

    The DarkNet architecture is commonly used for detection tasks. It is
    possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.

    Reference:
        - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
        - [YoloV3 implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        stackwise_blocks: integer, numbers of building blocks from the layer
            dark2 to layer dark5.
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone
    model = keras_cv.models.DarkNetBackbone.from_preset("darknet53_imagenet")
    output = model(input_data)

    # Randomly initialized backbone with a custom config
    model = DarkNetBackbone(
        stackwise_blocks=[2, 8, 8, 4],
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        stackwise_blocks,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        # stem
        pyramid_level_inputs = {}
        x = DarknetConvBlock(
            filters=32,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="stem_conv",
        )(x)
        pyramid_level_inputs[2] = x.node.layer.name
        x = ResidualBlocks(
            filters=64, num_blocks=1, name="stem_residual_block"
        )(x)
        pyramid_level_inputs[3] = x.node.layer.name

        # filters for the ResidualBlock outputs
        filters = [128, 256, 512, 1024]

        # layer_num is used for naming the residual blocks
        # (starts with dark2, hence 2)
        layer_num = 2

        for filter, block in zip(filters, stackwise_blocks):
            x = ResidualBlocks(
                filters=filter,
                num_blocks=block,
                name=f"dark{layer_num}_residual_block",
            )(x)
            layer_num += 1
            pyramid_level_inputs[layer_num + 1] = x.node.layer.name

        # remaining dark5 layers
        x = DarknetConvBlock(
            filters=512,
            kernel_size=1,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv1",
        )(x)
        pyramid_level_inputs[8] = x.node.layer.name
        x = DarknetConvBlock(
            filters=1024,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv2",
        )(x)
        pyramid_level_inputs[9] = x.node.layer.name
        x = SpatialPyramidPoolingBottleneck(
            512, activation="leaky_relu", name="dark5_spp"
        )(x)
        x = DarknetConvBlock(
            filters=1024,
            kernel_size=3,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv3",
        )(x)
        pyramid_level_inputs[10] = x.node.layer.name
        x = DarknetConvBlock(
            filters=512,
            kernel_size=1,
            strides=1,
            activation="leaky_relu",
            name="dark5_conv4",
        )(x)
        pyramid_level_inputs[11] = x.node.layer.name

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.pyramid_level_inputs = pyramid_level_inputs
        self.stackwise_blocks = stackwise_blocks
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_blocks": self.stackwise_blocks,
                "include_rescaling": self.include_rescaling,
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
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return copy.deepcopy(backbone_presets_with_weights)


ALIAS_DOCSTRING = """DarkNet model with {num_layers} layers.

    Although the DarkNet architecture is commonly used for detection tasks, it
    is possible to extract the intermediate dark2 to dark5 layers from the model
    for creating a feature pyramid Network.

    Reference:
        - [YoloV3 Paper](https://arxiv.org/abs/1804.02767)
        - [YoloV3 implementation](https://github.com/ultralytics/yolov3)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = DarkNet{num_layers}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


class DarkNet21Backbone(DarkNetBackbone):
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
        return DarkNetBackbone.from_preset("darknet21", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return {}


class DarkNet53Backbone(DarkNetBackbone):
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
        return DarkNetBackbone.from_preset("darknet53", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "darknet53_imagenet": copy.deepcopy(
                backbone_presets["darknet53_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include weights."""  # noqa: E501
        return cls.presets


setattr(DarkNet21Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=21))
setattr(DarkNet53Backbone, "__doc__", ALIAS_DOCSTRING.format(num_layers=53))
