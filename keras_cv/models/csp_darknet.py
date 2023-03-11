# Copyright 2022 The KerasCV Authors
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
import types

import tensorflow as tf
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
from keras_cv.models.weights import parse_weights


@keras.utils.regester_keras_serializable(package="keras_cv.models")
class CSPDarkNet(keras.Model):
    """This class Instantiates the CSPDarkNet architecture.

    Although the DarkNet architecture is commonly used for detection tasks, it is
    possible to extract the intermediate dark2 to dark5 layers from the model for
    creating a feature pyramid Network.

    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)
        - [YoloX implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        depth_multiplier: A float value used to calculate the base depth of the model
            this changes based the detection model being used.
        width_multiplier: A float value used to calculate the base width of the model
            this changes based the detection model being used.
        include_rescaling: bool ,whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: bool, whether to include the fully-connected layer at the top of
            the network.  If provided, `num_classes` must be provided.
        use_depthwise: a boolean value used to decide whether a depthwise conv block
            should be used over a regular darknet block. Defaults to False
        num_classes: optional int,optional number of classes to classify images into, only to be
            specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of the
                model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.
        name: (Optional) name to pass to the model.  Defaults to "CSPDarkNet".
    Returns:
        A `keras.Model` instance.
    """

    def __init__(
        self,
        depth_multiplier,
        width_multiplier,
        include_rescaling,
        include_top,
        use_depthwise=False,
        num_classes=None,
        weights=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        pooling=None,
        classifier_activation="softmax",
        name="CSPDarkNet",
        **kwargs,
    ):
        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                f"weights file to be loaded. Weights file not found at location{weights}"
            )

        if include_top and not num_classes:
            raise ValueError(
                "If `include_top` is True, you should specify `num_classes`. Received: "
                f"num_classes={num_classes}"
            )

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

        _backbone_level_outputs = {}
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
        _backbone_level_outputs[2] = x

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
        _backbone_level_outputs[3] = x

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
        _backbone_level_outputs[4] = x

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
        _backbone_level_outputs[5] = x

        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(
                num_classes,
                activation=classifier_activation,
                name="predictions",
            )(x)
        elif pooling == "avg":
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
        elif pooling == "max":
            x = layers.GlobalMaxPooling2D(name="max_pool")(x)

        model = keras.Model(inputs, x, name=name, **kwargs)
        model._backbone_level_outputs = _backbone_level_outputs
        # Bind the `to_backbone_model` method to the application model.
        model.as_backbone = types.MethodType(utils.as_backbone, model)

        if weights is not None:
            model.load_weights(weights)
        return model

    DEPTH_MULTIPLIERS = {
        "tiny": 0.33,
        "s": 0.33,
        "m": 0.67,
        "l": 1.00,
        "x": 1.33,
    }

    WIDTH_MULTIPLIERS = {
        "tiny": 0.375,
        "s": 0.50,
        "m": 0.75,
        "l": 1.00,
        "x": 1.25,
    }


BASE_DOCSTRING = """Instantiates the {name} architecture.

    The CSPDarkNet architectures are commonly used for detection tasks. It is
    possible to extract the intermediate dark2 to dark5 layers from the model for
    creating a feature pyramid Network.

    Reference:
        - [YoloV4 Paper](https://arxiv.org/abs/1804.02767)
        - [CSPNet Paper](https://arxiv.org/pdf/1911.11929)
        - [YoloX Paper](https://arxiv.org/abs/2107.08430)
        - [YoloX implementation](https://github.com/ultralytics/yolov3)
    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: whether or not to Rescale the inputs.If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        include_top: whether to include the fully-connected layer at the top of
            the network.  If provided, `num_classes` must be provided.
        use_depthwise: a boolean value used to decide whether a depthwise conv block
            should be used over a regular darknet block. Defaults to False
        num_classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g. 'imagenet/classification')
            (see available pre-trained weights in weights.py)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        pooling: optional pooling mode for feature extraction when `include_top`
            is `False`.
            - `None` means that the output of the model will be the 4D tensor output
                of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of the
                model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classifier_activation: A `str` or callable. The activation function to use
            on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top" layer.

        name: (Optional) name to pass to the model.  Defaults to "{name}".
    Returns:
        A `keras.Model` instance.
    """


def CSPDarkNetTiny(
    *,
    include_rescaling=None,
    include_top=None,
    use_depthwise=False,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="CSPDarkNetTiny",
    **kwargs,
):
    return CSPDarkNet(
        depth_multiplier=DEPTH_MULTIPLIERS["tiny"],
        width_multiplier=WIDTH_MULTIPLIERS["tiny"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        use_depthwise=use_depthwise,
        num_classes=num_classes,
        weights=parse_weights(weights, include_top, "cspdarknettiny"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def CSPDarkNetS(
    *,
    include_rescaling,
    include_top,
    use_depthwise=False,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="CSPDarkNetS",
    **kwargs,
):
    return CSPDarkNet(
        depth_multiplier=DEPTH_MULTIPLIERS["s"],
        width_multiplier=WIDTH_MULTIPLIERS["s"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        use_depthwise=use_depthwise,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def CSPDarkNetM(
    *,
    include_rescaling,
    include_top,
    use_depthwise=False,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="CSPDarkNetM",
    **kwargs,
):
    return CSPDarkNet(
        depth_multiplier=DEPTH_MULTIPLIERS["m"],
        width_multiplier=WIDTH_MULTIPLIERS["m"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        use_depthwise=use_depthwise,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def CSPDarkNetL(
    *,
    include_rescaling,
    include_top,
    use_depthwise=False,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="CSPDarkNetL",
    **kwargs,
):
    return CSPDarkNet(
        depth_multiplier=DEPTH_MULTIPLIERS["l"],
        width_multiplier=WIDTH_MULTIPLIERS["l"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        use_depthwise=use_depthwise,
        num_classes=num_classes,
        weights=parse_weights(weights, include_top, "cspdarknetl"),
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


def CSPDarkNetX(
    *,
    include_rescaling,
    include_top,
    use_depthwise=False,
    num_classes=None,
    weights=None,
    input_shape=(None, None, 3),
    input_tensor=None,
    pooling=None,
    classifier_activation="softmax",
    name="CSPDarkNetX",
    **kwargs,
):
    return CSPDarkNet(
        depth_multiplier=DEPTH_MULTIPLIERS["x"],
        width_multiplier=WIDTH_MULTIPLIERS["x"],
        include_rescaling=include_rescaling,
        include_top=include_top,
        use_depthwise=use_depthwise,
        num_classes=num_classes,
        weights=weights,
        input_shape=input_shape,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        name=name,
        **kwargs,
    )


setattr(CSPDarkNetTiny, "__doc__", BASE_DOCSTRING.format(name="CSPDarkNetTiny"))
setattr(CSPDarkNetS, "__doc__", BASE_DOCSTRING.format(name="CSPDarkNetS"))
setattr(CSPDarkNetM, "__doc__", BASE_DOCSTRING.format(name="CSPDarkNetM"))
setattr(CSPDarkNetL, "__doc__", BASE_DOCSTRING.format(name="CSPDarkNetL"))
setattr(CSPDarkNetX, "__doc__", BASE_DOCSTRING.format(name="CSPDarkNetX"))
