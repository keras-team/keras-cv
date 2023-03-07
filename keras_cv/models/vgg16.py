# Copyright 2022 The KerasCV Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""VGG16 model for KerasCV.
Reference:
  - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_cv.models import utils


def build_vgg_block(input_tensor,
                    num_layers,
                    kernel_size,
                    stride,
                    activation,
                    padding,
                    max_pool,
                    name,
                    ):
    """
    Builds VGG block
    Args:
        input_tensor: None (or) Tensor, input tensor to pass through network
        num_layers: int, number of CNN layers in the block
        kernel_size: int, kernel size of each CNN layer in block
        stride: int (or) tuple, stride for CNN layer in block
        activation: str (or) callable, activation function for each CNN layer in block
        padding: str (or) callable, padding function for each CNN layer in block
        max_pool: bool, whether or not to add MaxPooling2D layer at end of block.
        name: str, name of the block

    Returns:
        tf.Tensor
    """
    x = input_tensor
    for num in range(1, num_layers + 1):
        x = layers.Conv2D(kernel_size,
                          stride,
                          activation=activation,
                          padding=padding,
                          name=f"{name}_conv{str(num)}")(x)
    if max_pool:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f"{name}_pool")(x)
    return x


@keras.utils.register_keras_serializable(package="keras_cv.models")
class VGG16(keras.Model):
    """Instantiates the VGG16 architecture.
    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (ICLR 2015)
    This function returns a Keras VGG16 model.
    Args:
      include_rescaling: bool, whether or not to Rescale the inputs.If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
      include_top: bool, whether to include the 3 fully-connected
        layers at the top of the network. If provided, classes must be provided.
      classes: int, optional number of classes to classify images into, only to be
        specified if `include_top` is True.
      weights: os.PathLike or None, one of `None` (random initialization), or a pretrained weight file path.
      input_shape: tuple, optional shape tuple, defaults to (224, 224, 3).
      input_tensor: Tensor, optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.
      pooling: bool, Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classifier_activation:`str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      name: (Optional) name to pass to the model.  Defaults to "VGG16".
    Returns:
      A `keras.Model` instance.
    """
    def __init__(self,
                 input_tensor=None,
                 include_rescaling=True,
                 include_top=True,
                 classes=None,
                 weights=None,
                 input_shape=(224, 224, 3),
                 pooling=None,
                 classifier_activation="softmax",
                 name="VGG16",
                 **kwargs,
                 ):
        self.include_rescaling = include_rescaling
        self.include_top = include_top
        self.classes = classes
        self.input_tensor = input_tensor
        self.pooling = pooling
        self.classifier_activation = classifier_activation

        if weights and not tf.io.gfile.exists(weights):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                "weights file to be loaded. Weights file not found at location: {weights}"
            )

        if include_top and not classes:
            raise ValueError(
                "If `include_top` is True, you should specify `classes`. "
                f"Received: classes={classes}"
            )

        if include_top and pooling:
            raise ValueError(
                f"`pooling` must be `None` when `include_top=True`."
                f"Received pooling={pooling} and include_top={include_top}. "
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = layers.Rescaling(1 / 255.0)(x)

        x = build_vgg_block(input_tensor=x,
                            num_layers=2,
                            kernel_size=64,
                            stride=(3, 3),
                            activation='relu',
                            padding='same',
                            max_pool=True,
                            name='block1')

        x = build_vgg_block(input_tensor=x,
                            num_layers=2,
                            kernel_size=128,
                            stride=(3, 3),
                            activation='relu',
                            padding='same',
                            max_pool=True,
                            name='block2')

        x = build_vgg_block(input_tensor=x,
                            num_layers=3,
                            kernel_size=256,
                            stride=(3, 3),
                            activation='relu',
                            padding='same',
                            max_pool=True,
                            name='block3')

        x = build_vgg_block(input_tensor=x,
                            num_layers=3,
                            kernel_size=512,
                            stride=(3, 3),
                            activation='relu',
                            padding='same',
                            max_pool=True,
                            name='block4')

        x = build_vgg_block(input_tensor=x,
                            num_layers=3,
                            kernel_size=512,
                            stride=(3, 3),
                            activation='relu',
                            padding='same',
                            max_pool=True,
                            name='block5')

        if include_top:
            x = layers.Flatten(name="flatten")(x)
            x = layers.Dense(4096, activation="relu", name="fc1")(x)
            x = layers.Dense(4096, activation="relu", name="fc2")(x)
            x = layers.Dense(
                classes, activation=classifier_activation, name="predictions"
            )(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D()(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D()(x)

        super().__init__(inputs=inputs, outputs=x, name=name, **kwargs)
        if weights is not None:
            self.load_weights(weights)

    def get_config(self):
        return {
            "include_rescaling": self.include_rescaling,
            "include_top": self.include_top,
            "name": self.name,
            "input_shape": self.input_shape[1:],
            "input_tensor": self.input_tensor,
            "pooling": self.pooling,
            "classes": self.classes,
            "classifier_activation": self.classifier_activation,
            "trainables": self._trainable_weights,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
