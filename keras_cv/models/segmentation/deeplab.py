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

from keras_cv.layers.spatial_pyramid import SpatialPyramidPooling
from keras_cv.models.resnet_v2 import ResNetV2
from keras_cv.models.weights import parse_weights

BACKBONE_CONFIG = {
    "ResNet50V2": {
        "stackwise_filters": [64, 128, 256, 512],
        "stackwise_blocks": [3, 4, 6, 3],
        "stackwise_strides": [1, 2, 2, 2],
        "stackwise_dilations": [1, 1, 1, 2],
    }
}


class DeepLabV3(tf.keras.models.Model):
    """A segmentation model based on the DeepLab v3.

    Args:
        classes: int, the number of classes for the detection model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        include_rescaling: boolean, whether to Rescale the inputs. If set to True,
            inputs will be passed through a `Rescaling(1/255.0)` layer.
        backbone: an optional backbone network for the model. Can be a `tf.keras.layers.Layer`
            instance. The supported pre-defined backbone models are:
            1. "resnet50_v2", a ResNet50 V2 model
            Default to 'resnet50_v2'.
        backbone_weights: weights for the backbone model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        weights: weights for the complete DeepLabV3 model. one of `None` (random
            initialization), a pretrained weight file path, or a reference to
            pre-trained weights (e.g. 'imagenet/classification') (see available
            pre-trained weights in weights.py)
        decoder: an optional decoder network for segmentation model, e.g. FPN. The
            supported premade decoder is: "fpn". The decoder is called on
            the output of the backbone network to up-sample the feature output.
            Default to 'fpn'.
        segmentation_head: an optional `tf.keras.Layer` that predict the segmentation
            mask based on feature from backbone and feature from decoder.

    """

    def __init__(
        self,
        classes,
        include_rescaling,
        backbone,
        backbone_weights=None,
        spatial_pyramid_pooling=None,
        segmentation_head=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.classes = classes
        # ================== Backbone and weights. ==================
        if isinstance(backbone, str):
            supported_premade_backbone = [
                "resnet50_v2",
            ]
            if backbone not in supported_premade_backbone:
                raise ValueError(
                    "Supported premade backbones are: "
                    f'{supported_premade_backbone}, received "{backbone}"'
                )
            self._backbone_passed = backbone
            if backbone == "resnet50_v2":
                backbone = ResNetV2(
                    stackwise_filters=BACKBONE_CONFIG["ResNet50V2"][
                        "stackwise_filters"
                    ],
                    stackwise_blocks=BACKBONE_CONFIG["ResNet50V2"]["stackwise_blocks"],
                    stackwise_strides=BACKBONE_CONFIG["ResNet50V2"][
                        "stackwise_strides"
                    ],
                    stackwise_dilations=BACKBONE_CONFIG["ResNet50V2"][
                        "stackwise_dilations"
                    ],
                    include_rescaling=include_rescaling,
                    include_top=False,
                    name="resnet50v2",
                    weights=parse_weights(backbone_weights, False, "resnet50v2"),
                    pooling=None,
                    **kwargs,
                )

        else:
            # TODO(scottzhu): Might need to do more assertion about the model
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )
        self.backbone = backbone

        if spatial_pyramid_pooling is None:
            self.aspp = SpatialPyramidPooling(dilation_rates=[6, 12, 18])
        else:
            self.aspp = spatial_pyramid_pooling

        self._segmentation_head_passed = segmentation_head
        if segmentation_head is None:
            segmentation_head = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(
                        filters=256,
                        kernel_size=(1, 1),
                        padding="same",
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Activation("relu"),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Conv2D(
                        filters=classes,
                        kernel_size=(1, 1),
                        padding="same",
                        use_bias=False,
                        activation="softmax",
                        # Force the dtype of the classification head to float32 to avoid the NAN loss
                        # issue when used with mixed precision API.
                        dtype=tf.float32,
                    ),
                ]
            )
        self.segmentation_head = segmentation_head

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]
        feature_map_shape = self.backbone.compute_output_shape(input_shape)
        self.up_layer = tf.keras.layers.UpSampling2D(
            size=(height // feature_map_shape[1], width // feature_map_shape[2]),
            interpolation="bilinear",
        )

    def call(self, inputs, training=None):
        feature_map = self.backbone(inputs, training=training)
        output = self.aspp(feature_map, training=training)
        output = self.up_layer(output, training=training)
        output = self.segmentation_head(output, training=training)
        return output

    # TODO(tanzhenyu): consolidate how regularization should be applied to KerasCV.
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

    def get_config(self):
        config = {
            "classes": self.classes,
            "backbone": self._backbone_passed,
            "decoder": self._decoder_passed,
            "segmentation_head": self._segmentation_head_passed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
