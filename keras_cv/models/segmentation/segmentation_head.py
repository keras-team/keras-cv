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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="keras_cv")
class SegmentationHead(layers.Layer):
    """Prediction head for the segmentation model

    The head will take the output from decoder (eg FPN or ASPP), and produce a
    segmentation mask (pixel level classifications) as the output for the model.

    Args:
        num_classes: int, number of output classes for the prediction. This
            should include all the classes (e.g. background) for the model to
            predict.
        convolutions: int, number of `Conv2D` layers that are stacked before the
            final classification layer, defaults to 2.
        filters: int, number of filter/channels for the conv2D layers.
            Defaults to 256.
        activations: str or function, activation functions between the conv2D
            layers and the final classification layer, defaults to `"relu"`.
        output_scale_factor: int, or a pair of ints. Factor for upsampling the
            output mask. This is useful to scale the output mask back to same
            size as the input image. When single int is provided, the mask will
            be scaled with same ratio on both width and height. When a pair of
            ints are provided, they will be parsed as `(height_factor,
            width_factor)`. Defaults to `None`, which means no resize will
            happen to the output mask tensor.
        kernel_size: int, the kernel size to be used in each of the
            convolutional blocks, defaults to 3.
        use_bias: boolean, whether to use bias or not in each of the
            convolutional blocks, defaults to False since the blocks use
            `BatchNormalization` after each convolution, rendering bias
            obsolete.
        activation: str or function, activation to apply in the classification
            layer (output of the head), defaults to `"softmax"`.

    Examples:

    ```python
    # Mimic a FPN output dict
    p3 = tf.ones([2, 32, 32, 3])
    p4 = tf.ones([2, 16, 16, 3])
    p5 = tf.ones([2, 8, 8, 3])
    inputs = {3: p3, 4: p4, 5: p5}

    head = SegmentationHead(num_classes=11)

    output = head(inputs)
    # output tensor has shape [2, 32, 32, 11]. It has the same resolution as
    the p3.
    ```
    """

    def __init__(
        self,
        num_classes,
        convolutions=2,
        filters=256,
        activations="relu",
        dropout=0.0,
        kernel_size=3,
        activation="softmax",
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.convolutions = convolutions
        self.filters = filters
        self.activations = activations
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation

        self._conv_layers = []
        self._bn_layers = []
        for i in range(self.convolutions):
            conv_name = "segmentation_head_conv_{}".format(i)
            self._conv_layers.append(
                keras.layers.Conv2D(
                    name=conv_name,
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding="same",
                    use_bias=self.use_bias,
                )
            )
            norm_name = "segmentation_head_norm_{}".format(i)
            self._bn_layers.append(
                keras.layers.BatchNormalization(name=norm_name)
            )

        self._classification_layer = keras.layers.Conv2D(
            name="segmentation_output",
            filters=self.num_classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation=self.activation,
            # Force the dtype of the classification head to float32 to avoid the
            # NAN loss issue when used with mixed precision API.
            dtype=tf.float32,
        )

        self.dropout_layer = keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        """Forward path for the segmentation head.

        For now, it accepts the output from the decoder only, which is a dict
        with int key and tensor as value (level-> processed feature output). The
        head will use the lowest level of feature output as the input for the
        head.
        """
        if not isinstance(inputs, dict):
            raise ValueError(
                f"Expect inputs to be a dict. Received instead inputs={inputs}"
            )

        lowest_level = next(iter(sorted(inputs)))
        x = inputs[lowest_level]
        for conv_layer, bn_layer in zip(self._conv_layers, self._bn_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = keras.activations.get(self.activations)(x)
            if self.dropout:
                x = self.dropout_layer(x)
        return self._classification_layer(x)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "convolutions": self.convolutions,
            "filters": self.filters,
            "activations": self.activations,
            "dropout": self.dropout,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "activation": self.activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
