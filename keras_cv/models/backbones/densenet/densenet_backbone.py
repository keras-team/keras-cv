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

"""DenseNet models for KerasCV.

Reference:
  - [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  - [Based on the Original keras.applications DenseNet](https://github.com/keras-team/keras/blob/master/keras/applications/densenet.py)
"""  # noqa: E501

import copy

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.densenet.densenet_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.densenet.densenet_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty

BN_AXIS = 3
BN_EPSILON = 1.001e-5

BASE_DOCSTRING = """Instantiates the {name} architecture.

    Reference:
        - [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

    This function returns a Keras {name} model.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        num_classes: optional int, number of classes to classify images into
            (only to be specified if `include_top` is `True`).
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g.
            'imagenet/classification')(see available pre-trained weights in
            weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: (Optional) name to pass to the model, defaults to "{name}".
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.

    Returns:
      A `keras.Model` instance.
"""  # noqa: E501


def apply_dense_block(x, blocks, name=None):
    """A dense block.

    Args:
      blocks: int, number of building blocks.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing an apply_dense_block.
    """
    if name is None:
        name = f"dense_block_{backend.get_uid('dense_block')}"

    for i in range(blocks):
        x = apply_conv_block(x, 32, name=f"{name}_block_{i}")
    return x


def apply_transition_block(x, reduction, name=None):
    """A transition block.

    Args:
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing an
      apply_transition_block.
    """
    if name is None:
        name = f"transition_block_{backend.get_uid('transition_block')}"

    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[BN_AXIS] * reduction),
        1,
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.AveragePooling2D(2, strides=2, name=f"{name}_pool")(x)
    return x


def apply_conv_block(x, growth_rate, name=None):
    """A building block for a dense block.

    Args:
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      a function that takes an input Tensor representing a apply_conv_block.
    """
    if name is None:
        name = f"conv_block_{backend.get_uid('conv_block')}"

    x1 = x
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_0_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_0_relu")(x)
    x = layers.Conv2D(
        4 * growth_rate, 1, use_bias=False, name=f"{name}_1_conv"
    )(x)
    x = layers.BatchNormalization(
        axis=BN_AXIS, epsilon=BN_EPSILON, name=f"{name}_1_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_1_relu")(x)
    x = layers.Conv2D(
        growth_rate,
        3,
        padding="same",
        use_bias=False,
        name=f"{name}_2_conv",
    )(x)
    x = layers.Concatenate(axis=BN_AXIS, name=f"{name}_concat")([x1, x])
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class DenseNetBackbone(Backbone):
    """Instantiates the DenseNet architecture.

    Reference:
        - [Densely Connected Convolutional Networks (CVPR 2017)](https://arxiv.org/abs/1608.06993)

    This function returns a Keras DenseNet model.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        blocks: numbers of building blocks for the four dense layers.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        include_top: bool, whether to include the fully-connected layer at the
            top of the network. If provided, `num_classes` must be provided.
        num_classes: optional int, number of classes to classify images into
            (only to be specified if `include_top` is `True`).
        weights: one of `None` (random initialization), a pretrained weight file
            path, or a reference to pre-trained weights (e.g.
            'imagenet/classification')(see available pre-trained weights in
            weights.py)
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        name: (Optional) name to pass to the model, defaults to "DenseNet".
        classifier_activation: A `str` or callable. The activation function to
            use on the "top" layer. Ignored unless `include_top=True`. Set
            `classifier_activation=None` to return the logits of the "top"
            layer.

    Returns:
      A `keras.Model` instance.
    """  # noqa: E501

    def __init__(
        self,
        *,
        blocks,
        include_rescaling,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        x = inputs
        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = layers.Conv2D(
            64, 7, strides=2, use_bias=False, padding="same", name="conv1/conv"
        )(x)
        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="conv1/bn"
        )(x)
        x = layers.Activation("relu", name="conv1/relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

        pyramid_level_inputs = {}
        x = apply_dense_block(x, blocks[0], name="conv2")
        pyramid_level_inputs[2] = x.node.layer.name
        x = apply_transition_block(x, 0.5, name="pool2")
        pyramid_level_inputs[3] = x.node.layer.name
        x = apply_dense_block(x, blocks[1], name="conv3")
        pyramid_level_inputs[4] = x.node.layer.name
        x = apply_transition_block(x, 0.5, name="pool3")
        pyramid_level_inputs[5] = x.node.layer.name
        x = apply_dense_block(x, blocks[2], name="conv4")
        pyramid_level_inputs[6] = x.node.layer.name
        x = apply_transition_block(x, 0.5, name="pool4")
        pyramid_level_inputs[7] = x.node.layer.name
        x = apply_dense_block(x, blocks[3], name="conv5")
        pyramid_level_inputs[8] = x.node.layer.name

        x = layers.BatchNormalization(
            axis=BN_AXIS, epsilon=BN_EPSILON, name="bn"
        )(x)
        x = layers.Activation("relu", name="relu")(x)

        # Create model.
        super().__init__(inputs=inputs, outputs=x, **kwargs)

        # All references to `self` below this line

        self.blocks = blocks
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "blocks": self.blocks,
                "include_rescaling": self.include_rescaling,
                # Remove batch dimension from `input_shape`
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


setattr(
    DenseNet121Backbone, "__doc__", BASE_DOCSTRING.format(name="DenseNet121")
)
setattr(
    DenseNet169Backbone, "__doc__", BASE_DOCSTRING.format(name="DenseNet169")
)
setattr(
    DenseNet201Backbone, "__doc__", BASE_DOCSTRING.format(name="DenseNet201")
)
