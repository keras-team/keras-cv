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

"""VGG16 model for KerasCV.
Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
    (ICLR 2015)
"""  # noqa: E501
import copy

from tensorflow import keras
from tensorflow.keras import layers

from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.vgg16.vgg16_backbone_presets import (
    backbone_presets,
)
from keras_cv.utils.python_utils import classproperty


def apply_vgg_block(
    x,
    num_layers,
    filters,
    kernel_size,
    activation,
    padding,
    max_pool,
    name,
):
    """
    Applies VGG block
    Args:
        x: Tensor, input tensor to pass through network
        num_layers: int, number of CNN layers in the block
        filters: int, filter size of each CNN layer in block
        kernel_size: int (or) tuple, kernel size for CNN layer in block
        activation: str (or) callable, activation function for each CNN layer in
            block
        padding: str (or) callable, padding function for each CNN layer in block
        max_pool: bool, whether to add MaxPooling2D layer at end of block
        name: str, name of the block

    Returns:
        tf.Tensor
    """
    for num in range(1, num_layers + 1):
        x = layers.Conv2D(
            filters,
            kernel_size,
            activation=activation,
            padding=padding,
            name=f"{name}_conv{str(num)}",
        )(x)
    if max_pool:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f"{name}_pool")(x)
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class VGG16Backbone(Backbone):
    """Instantiates the VGG16 architecture.

    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
        (ICLR 2015)

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        input_shape: tuple, optional shape tuple, defaults to (224, 224, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.

    Examples:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone with a custom config
    model = VGG16Backbone(
        include_rescaling=False,
    )
    output = model(input_data)
    ```
    """  # noqa: E501

    def __init__(
        self,
        *,
        include_rescaling,
        input_tensor=None,
        input_shape=(224, 224, 3),
        **kwargs,
    ):
        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = apply_vgg_block(
            x=x,
            num_layers=2,
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block1",
        )

        x = apply_vgg_block(
            x=x,
            num_layers=2,
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block2",
        )

        x = apply_vgg_block(
            x=x,
            num_layers=3,
            filters=256,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block3",
        )

        x = apply_vgg_block(
            x=x,
            num_layers=3,
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block4",
        )

        x = apply_vgg_block(
            x=x,
            num_layers=3,
            filters=512,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
            max_pool=True,
            name="block5",
        )

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
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
