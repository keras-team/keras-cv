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
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="keras_cv")
class SegmentationHead(layers.Layer):
    """Prediction head for the segmentation model

    The head will take the output from decoder (eg FPN or ASPP), and produce a
    segmentation mask (pixel level classifications) as the output for the model.

    Args:
        classes: int, the number of output classes for the prediction. This should
            include all the classes (eg background) for the model to predict.
        convs: int, the number of conv2D layers that are stacked before the final
            classification layer. Default to 2.
        filters: int, the number of filter/channels for the the conv2D layers. Default
            to 256.
        activations: str or 'tf.keras.activations', activation functions between the
            conv2D layers and the final classification layer. Default to 'relu'
        output_scale_factor: int, or a pair of ints, for upsample the output mask.
            This is useful to scale the output mask back to same size as the input
            image. When single int is provided, the mask will be scaled with same
            ratio on both width and height. When a pair of ints are provided, they will
            be parsed as (height_factor, width_factor). Default to None, which means
            no resize will happen to the output mask tensor.
        kernel_size: default 3; the kernel_size to be used in each of the `convs` blocks
        use_bias: default False; whether to use bias or not in each of the `convs` blocks
                Defaults to none since the blocks use `BatchNormalization` after each conv, rendering
                bias obsolete
        activation: default 'softmax', the activation to apply in the classification
            layer (output of the head)

    Sample code
    ```python
    # Mimic a FPN output dict
    p3 = tf.ones([2, 32, 32, 3])
    p4 = tf.ones([2, 16, 16, 3])
    p5 = tf.ones([2, 8, 8, 3])
    inputs = {3: p3, 4: p4, 5: p5}

    head = SegmentationHead(classes=11)

    output = head(inputs)
    # output tensor has shape [2, 32, 32, 11]. It has the same resolution as the p3.
    ```
    """

    def __init__(
        self,
        classes,
        convs=2,
        filters=256,
        activations="relu",
        dropout=0.0,
        kernel_size=3,
        output_scale_factor=None,
        activation="softmax",
        conv_type="depthwise_separable_conv",
        use_bias=False,
        **kwargs,
    ):
        """
        Args:
            classes: the number of possible classes for the segmentation map
            convs: default 2; the number of conv blocks to use in the head (conv2d-batch_norm-activation blocks)
            filters: default 256; the number of filters in each Conv2D layer
            activations: default 'relu'; the activation to apply in conv blocks
            dropout: default 0.0; the dropout to apply between each conv block
            kernel_size: default 3; the kernel_size to be used in each of the `convs` blocks
            use_bias: default False; whether to use bias or not in each of the `convs` blocks
                Defaults to none since the blocks use `BatchNormalization` after each conv, rendering
                bias obsolete
            activation: default 'softmax', the activation to apply in the classification
                layer (output of the head)
            **kwargs:
        """
        super().__init__(**kwargs)
        self.classes = classes
        self.convs = convs
        self.filters = filters
        self.activations = activations
        self.output_scale_factor = output_scale_factor
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation
        self.conv_type = conv_type

        self._conv_layers = []
        self._bn_layers = []
        for i in range(self.convs):
            conv_name = "segmentation_head_conv_{}".format(i)
            if self.conv_type == "conv2d":
                self._conv_layers.append(
                    tf.keras.layers.Conv2D(
                        name=conv_name,
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding="same",
                        use_bias=self.use_bias,
                    )
                )
            else:
                self._conv_layers.append(
                    tf.keras.layers.SeparableConv2D(
                        name=conv_name,
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        padding="same",
                        use_bias=self.use_bias,
                    )
                )

            norm_name = "segmentation_head_norm_{}".format(i)
            self._bn_layers.append(tf.keras.layers.BatchNormalization(name=norm_name))

        self._classification_layer = tf.keras.layers.Conv2D(
            name="segmentation_output",
            filters=self.classes,
            kernel_size=1,
            use_bias=False,
            padding="same",
            activation=self.activation,
            # Force the dtype of the classification head to float32 to avoid the NAN loss
            # issue when used with mixed precision API.
            dtype=tf.float32,
        )

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        x = inputs
        for conv_layer, bn_layer in zip(self._conv_layers, self._bn_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = tf.keras.activations.get(self.activations)(x)

        if self.output_scale_factor is not None:
            x = tf.keras.layers.UpSampling2D(self.output_scale_factor)(x)

        if self.dropout:
            x = self.dropout_layer(x)

        x = self._classification_layer(x)
        return x

    def get_config(self):
        config = {
            "classes": self.classes,
            "convs": self.convs,
            "filters": self.filters,
            "activations": self.activations,
            "dropout": self.dropout,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "activation": self.activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
