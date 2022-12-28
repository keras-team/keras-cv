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
from keras_cv.models.weights import parse_weights


@keras.utils.register_keras_serializable(package="keras_cv")
class DeepLabV3Plus(keras.Model):
    """
    A segmentation model based on the DeepLabV3Plus model.
    References:
        - [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        backbone: a backbone for the model, expected to be a KerasCV model.
            Typically ResNet50V2 or ResNet101V2. Default 'low_level_feature_layer' assumes
            either and uses 'v2_stack_1_block4_1_relu' for them by default.
        low_level_feature_layer: the layer name for the low-level features to use for encoding/decoding
            spatial information for the supplied backbone. The high-level activations come from the last layer in the model.
        weights: weights for the complete DeepLabV3Plus model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        spatial_pyramid_pooling: also known as Atrous Spatial Pyramid Pooling (ASPP).
            Performs spatial pooling on different spatial levels in the pyramid, with
            dilation.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.
        segmentation_head_activation: default 'softmax', the activation layer to apply after
            the segmentation head. Should be synchronized with the backbone's final activation
    """

    def __init__(
        self,
        classes,
        backbone,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        segmentation_head_activation="softmax",
        input_shape=(None, None, 3),
        input_tensor=None,
        low_level_feature_layer=None,
        weights=None,
        **kwargs,
    ):

        if not isinstance(backbone, tf.keras.layers.Layer):
            raise ValueError(
                "Backbone need to be a `tf.keras.layers.Layer`, " f"received {backbone}"
            )

        if weights and not tf.io.gfile.exists(
            parse_weights(weights, True, "deeplabv3plus")
        ):
            raise ValueError(
                "The `weights` argument should be either `None` or the path to the "
                f"weights file to be loaded. Weights file not found at location: {weights}"
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = backbone.input_shape[1:]
            inputs = layers.Input(tensor=backbone.input, shape=input_shape)

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3Plus are `None`."
            )

        x = inputs
        high_level = backbone(x)

        if low_level_feature_layer is None:
            if "resnet" in backbone.name:
                low_level = backbone.get_layer("v2_stack_1_block4_1_relu").output
            else:
                raise ValueError(
                    "You have to specify the name of the low-level layer in the "
                    "model used to extract low-level features."
                )
        else:
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )
            low_level = backbone.get_layer(low_level_feature_layer).output

        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(dilation_rates=[6, 12, 18])

        output = spatial_pyramid_pooling(high_level)
        output = tf.keras.layers.UpSampling2D(
            size=(4, 4),
            interpolation="bilinear",
        )(output)

        low_level = layers.Conv2D(
            filters=256,
            kernel_size=1,
            name="project_conv_bn_act",
            use_bias=False,
        )(low_level)
        low_level = layers.BatchNormalization()(low_level)
        low_level = layers.Activation("relu")(low_level)

        output = layers.Concatenate()([output, low_level])

        if segmentation_head is None:
            segmentation_head = SegmentationHead(
                classes=classes,
                name="segmentation_head",
                conv_type="depthwise_separable_conv",
                output_scale_factor=4,
                filters=256,
                convs=2,
                dropout=0.2,
                kernel_size=3,
            )

        output = segmentation_head(output)

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
        self.low_level_feature_layer = low_level_feature_layer

    def get_config(self):
        return {
            "classes": self.classes,
            "backbone": self.backbone,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
            "low_level_feature_layer": self.low_level_feature_layer,
        }

    def compile(self, weight_decay=0.0001, **kwargs):
        self.weight_decay = weight_decay
        super().compile(**kwargs)

    def train_step(self, data):
        images, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            total_loss = self.compute_loss(images, y_true, y_pred, sample_weight)
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(self.weight_decay * tf.nn.l2_loss(var))
                l2_loss = tf.math.add_n(reg_losses)
                total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(images, y_true, y_pred, sample_weight=sample_weight)


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
            "dropout": self.dropout,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "activation": self.activation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
