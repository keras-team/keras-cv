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

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models import utils

BACKBONE_CONFIG = {
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "stackwise_dilations": [1, 1, 1, 2],
    }
}


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3(keras.Model):
    """
    A segmentation model based on the DeepLab v3.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        backbone: an optional backbone network for the model. Should be a KerasCV model.
        weights: weights for the complete DeepLabV3 model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        spatial_pyramid_pooling: also known as Atrous Spatial Pyramid Pooling (ASPP).
            Performs spatial pooling on different spatial levels in the pyramid, with
            dilation.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
    """

    def __init__(
        self,
        classes,
        backbone,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation=None,
        input_shape=(None, None, 3),
        input_tensor=None,
        **kwargs,
    ):

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        height = input_shape[0]
        width = input_shape[1]
        channels = input_shape[2]

        x = inputs

        if not isinstance(backbone, tf.keras.layers.Layer):
            raise ValueError(
                "Backbone need to be a `tf.keras.layers.Layer`, " f"received {backbone}"
            )

        feature_map = backbone(x)
        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(dilation_rates=[6, 12, 18])

        output = spatial_pyramid_pooling(feature_map)
        output = tf.keras.layers.UpSampling2D(
            size=(height // feature_map.shape[1], width // feature_map.shape[2]),
            interpolation="bilinear",
        )(output)

        if segmentation_head is None:
            segmentation_head = SegmentationHead(
                classes=classes,
                name="segmentation_head",
                convs=1,
                dropout=0.2,
                kernel_size=1,
                output_activation=segmentation_head_activation,
            )

        # Segmentation head expects a multiple-level output dictionary
        output = segmentation_head({1: output})

        super().__init__(
            inputs={
                "inputs": inputs,
            },
            outputs={
                "output": output,
            },
            **kwargs,
        )

        # All references to `self` below this line
        self.classes = classes
        self.backbone = backbone
        self.spatial_pyramid_pooling = spatial_pyramid_pooling
        self.segmentation_head = segmentation_head
        self.segmentation_head_activation = segmentation_head_activation

    def get_config(self):
        return {
            "classes": self.classes,
            "backbone": self.backbone,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
        }


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
        output_activation: default 'softmax', the activation to apply in the classification
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
        output_scale_factor=None,
        dropout=0.0,
        kernel_size=3,
        output_activation="softmax",
        use_bias=False,
        **kwargs,
    ):
        """
        Args:
            classes: the number of possible classes for the segmentation map
            convs: default 2; the number of conv blocks to use in the head (conv2d-batch_norm-activation blocks)
            filters: default 256; the number of filters in each Conv2D layer
            activations: default 'relu'; the activation to apply in conv blocks
            output_scale_factor: default None; the scale to apply in the UpSampling call before the output
            dropout: default 0.0; the dropout to apply between each conv block
            kernel_size: default 3; the kernel_size to be used in each of the `convs` blocks
            use_bias: default False; whether to use bias or not in each of the `convs` blocks
                Defaults to none since the blocks use `BatchNormalization` after each conv, rendering
                bias obsolete
            output_activation: default 'softmax', the activation to apply in the classification
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
        self.output_activation = output_activation

        self._conv_layers = []
        self._bn_layers = []
        for i in range(self.convs):
            conv_name = "segmentation_head_conv_{}".format(i)
            self._conv_layers.append(
                tf.keras.layers.Conv2D(
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
            padding="same",
            activation=self.output_activation,
            # Force the dtype of the classification head to float32 to avoid the NAN loss
            # issue when used with mixed precision API.
            dtype=tf.float32,
        )

        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs):
        """Forward path for the segmentation head.

        For now, it accepts the output from the decoder only, which is a dict with int
        key and tensor as value (level-> processed feature output). The head will use the
        lowest level of feature output as the input for the head.
        """
        if not isinstance(inputs, dict):
            raise ValueError(f"Expect the inputs to be a dict, but received {inputs}")

        lowest_level = next(iter(sorted(inputs)))
        x = inputs[lowest_level]
        for conv_layer, bn_layer in zip(self._conv_layers, self._bn_layers):
            x = conv_layer(x)
            x = bn_layer(x)
            x = tf.keras.activations.get(self.activations)(x)
            if self.dropout:
                x = self.dropout_layer(x)

        if self.output_scale_factor is not None:
            x = tf.keras.layers.UpSampling2D(self.output_scale_factor)(x)
        x = self._classification_layer(x)
        return x

    def get_config(self):
        config = {
            "classes": self.classes,
            "convs": self.convs,
            "filters": self.filters,
            "activations": self.activations,
            "output_scale_factor": self.output_scale_factor,
            "dropout": self.dropout,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
