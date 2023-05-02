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
from keras_cv.models.backbones.csp_darknet import csp_darknet_utils
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.csp_darknet.csp_darknet_backbone_presets import (
    backbone_presets_with_weights,
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
        include_focus: Boolean, whether to use `Focus` layer at the beginning of
            the backbone. Defaults to `True`.
        use_depthwise: bool, whether a `DarknetConvBlockDepthwise` should be
            used over a `DarknetConvBlock`, defaults to False.
        darknet_padding: String, the padding used in the `Conv2D` layers in the
            `DarknetConvBlock`s. Defaults to `"same"`.
        darknet_zero_padding: Boolean, whether to use `ZeroPadding2D` layer at
            the beginning of each `DarknetConvBlock`. The zero padding will only
            be applied when `kernel_size` > 1. Defaults to `False`.
        darknet_bn_momentum: Float, momentum for the moving average for the
            `BatchNormalization` layers in the `DarknetConvBlock`s. Defaults to
            `0.99`.
        stem_stride: The stride to use of the `Conv2D` layers in the stem part
            of the backbone. Defaults to `1`.
        csp_wide_stem: Boolean, in the CSP blocks, whether to combine the first
            two `DarknetConvBlock`s into one with more filters and split the
            outputs to two tensors.  Defaults to `False`.
        csp_kernel_sizes: A list of integers of length 2. The kernel sizes of the
            bottleneck layers in the CSP blocks. Defaults to `[1, 3]`.
        csp_concat_bottleneck_outputs: Boolean, in the CSP blocks, whether to
            concatenate the outputs of all the bottleneck blocks as the output
            for the next layer. If `False`, only the output of the last
            bottleneck block is used.  Defaults to `False`.
        csp_always_residual: Boolean, whether to always use residual connections
            for the CSP blocks. If `False`, residual connections will be applied
            to all CSP blocks but the last one. Defautls to `False`.
        spp_after_csp=False,
        spp_pool_sizes=(5, 9, 13),
        spp_sequential_pooling=False,
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
    model = CSPDarkNetBackbone(
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
        include_focus=True,
        use_depthwise=False,
        darknet_padding="same",
        darknet_zero_padding=False,
        darknet_bn_momentum=0.99,
        stem_stride=1,
        csp_wide_stem=False,
        csp_kernel_sizes=[1, 3],
        csp_concat_bottleneck_outputs=False,
        csp_always_residual=False,
        spp_after_csp=False,
        spp_pool_sizes=(5, 9, 13),
        spp_sequential_pooling=False,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        apply_conv_block = (
            csp_darknet_utils.apply_darknet_conv_block_depthwise
            if use_depthwise
            else csp_darknet_utils.apply_darknet_conv_block
        )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0, name="rescaling")(x)

        # stem
        if include_focus:
            x = csp_darknet_utils.apply_focus(x, name="stem_focus")

        stem_width = stackwise_channels[0]

        x = apply_conv_block(
            x,
            stem_width // 2,
            kernel_size=3,
            strides=stem_stride,
            padding=darknet_padding,
            use_zero_padding=darknet_zero_padding,
            batch_norm_momentum=darknet_bn_momentum,
            name="stem_conv",
        )

        pyramid_level_inputs = {}
        for index, (channels, depth) in enumerate(
            zip(stackwise_channels, stackwise_depth)
        ):
            x = apply_conv_block(
                x,
                channels,
                kernel_size=3,
                strides=2,
                padding=darknet_padding,
                use_zero_padding=darknet_zero_padding,
                batch_norm_momentum=darknet_bn_momentum,
                name=f"dark_{index + 2}_conv",
            )

            def spp(x):
                return csp_darknet_utils.apply_spatial_pyramid_pooling_bottleneck(  # noqa: E501
                    x,
                    channels,
                    hidden_filters=channels // 2,
                    kernel_sizes=spp_pool_sizes,
                    padding=darknet_padding,
                    use_zero_padding=darknet_zero_padding,
                    batch_norm_momentum=darknet_bn_momentum,
                    sequential_pooling=spp_sequential_pooling,
                    name=f"dark_{index + 2}_spp",
                )

            def csp(x):
                return csp_darknet_utils.apply_cross_stage_partial(
                    x,
                    channels,
                    num_bottlenecks=depth,
                    use_depthwise=use_depthwise,
                    residual=(
                        csp_always_residual
                        or (index != len(stackwise_depth) - 1)
                    ),
                    wide_stem=csp_wide_stem,
                    kernel_sizes=csp_kernel_sizes,
                    concat_bottleneck_outputs=csp_concat_bottleneck_outputs,
                    padding=darknet_padding,
                    use_zero_padding=darknet_zero_padding,
                    batch_norm_momentum=darknet_bn_momentum,
                    name=f"dark_{index + 2}_csp",
                )

            if index != len(stackwise_depth) - 1:
                x = csp(x)
            elif spp_after_csp:
                x = csp(x)
                x = spp(x)
            else:
                x = spp(x)
                x = csp(x)

            pyramid_level_inputs[index + 2] = x.node.layer.name

        super().__init__(inputs=inputs, outputs=x, **kwargs)
        self.pyramid_level_inputs = pyramid_level_inputs

        self.stackwise_channels = stackwise_channels
        self.stackwise_depth = stackwise_depth
        self.include_rescaling = include_rescaling
        self.use_depthwise = use_depthwise
        self.include_focus = include_focus
        self.darknet_bn_momentum = darknet_bn_momentum
        self.darknet_zero_padding = darknet_zero_padding
        self.darknet_padding = darknet_padding
        self.stem_stride = stem_stride
        self.csp_wide_stem = csp_wide_stem
        self.csp_kernel_sizes = csp_kernel_sizes
        self.csp_concat_bottleneck_outputs = csp_concat_bottleneck_outputs
        self.csp_always_residual = csp_always_residual
        self.spp_after_csp = spp_after_csp
        self.spp_pool_sizes = spp_pool_sizes
        self.spp_sequential_pooling = spp_sequential_pooling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stackwise_channels": self.stackwise_channels,
                "stackwise_depth": self.stackwise_depth,
                "include_rescaling": self.include_rescaling,
                "use_depthwise": self.use_depthwise,
                "include_focus": self.include_focus,
                "darknet_bn_momentum": self.darknet_bn_momentum,
                "darknet_zero_padding": self.darknet_zero_padding,
                "darknet_padding": self.darknet_padding,
                "stem_stride": self.stem_stride,
                "csp_wide_stem": self.csp_wide_stem,
                "csp_kernel_sizes": self.csp_kernel_sizes,
                "csp_concat_bottleneck_outputs": self.csp_concat_bottleneck_outputs,  # noqa: E501
                "csp_always_residual": self.csp_always_residual,
                "spp_after_csp": self.spp_after_csp,
                "spp_pool_sizes": self.spp_pool_sizes,
                "spp_sequential_pooling": self.spp_sequential_pooling,
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
