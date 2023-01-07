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
from keras_cv.models.segmentation.segmentation_head import SegmentationHead
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
        layer_names: the layer names for the low-level and high-level features to use for encoding/decoding
            spatial information for the supplied backbone, respectively.
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
        layer_names=(None, None),
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
        x = inputs

        if input_shape[0] is None and input_shape[1] is None:
            input_shape = backbone.input_shape[1:]

        if input_shape[0] is None and input_shape[1] is None:
            raise ValueError(
                "Input shapes for both the backbone and DeepLabV3Plus are `None`."
            )

        if layer_names == (None, None):
            if "res" in backbone.name:
                low_level_output = backbone.get_layer("v2_stack_0_block3_out").output
                high_level_output = backbone.get_layer("v2_stack_3_block3_out").output
            else:
                raise ValueError(
                    "Cannot default low-level and high-level layer names with a custom backbone."
                    f"Passed layer_names: {layer_names}"
                )
        else:
            low_level_output = backbone.get_layer(layer_names[0]).output
            high_level_output = backbone.get_layer(layer_names[1]).output

        backbone_outputs = {
            "low_level": low_level_output,
            "high_level": high_level_output,
        }
        backbone = tf.keras.Model(backbone.input, backbone_outputs)
        backbone_outputs = backbone(x)

        low_level = backbone_outputs["low_level"]
        high_level = backbone_outputs["high_level"]

        if spatial_pyramid_pooling is None:
            spatial_pyramid_pooling = SpatialPyramidPooling(
                dilation_rates=[6, 12, 18], dropout=0.5
            )

        output = spatial_pyramid_pooling(high_level)
        output = tf.keras.layers.UpSampling2D(
            size=(4, 4),
            interpolation="bilinear",
        )(output)

        low_level = layers.Conv2D(
            filters=48,
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
                conv_type="conv2d",
                output_scale_factor=4,
                filters=256,
                convs=2,
                dropout=0.2,
                kernel_size=3,
                activation=segmentation_head_activation,
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
        self.layer_names = layer_names

    def get_config(self):
        return {
            "classes": self.classes,
            "backbone": self.backbone,
            "spatial_pyramid_pooling": self.spatial_pyramid_pooling,
            "segmentation_head": self.segmentation_head,
            "segmentation_head_activation": self.segmentation_head_activation,
            "layer_names": self.layer_names,
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
