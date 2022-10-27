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

import keras_cv


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
        backbone="resnet50_v2",
        decoder="fpn",
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
                backbone = keras_cv.models.ResNet50V2(
                    include_rescaling=include_rescaling, include_top=False
                )
                backbone = backbone.as_backbone()
                self.backbone = backbone
        else:
            # TODO(scottzhu): Might need to do more assertion about the model
            if not isinstance(backbone, tf.keras.layers.Layer):
                raise ValueError(
                    "Backbone need to be a `tf.keras.layers.Layer`, "
                    f"received {backbone}"
                )
            self.backbone = backbone

        # ================== decoder ==================
        if isinstance(decoder, str):
            # TODO(scottzhu): Add ASPP decoder.
            supported_premade_decoder = ["fpn"]
            if decoder not in supported_premade_decoder:
                raise ValueError(
                    "Supported premade decoder are: "
                    f'{supported_premade_decoder}, received "{decoder}"'
                )
            self._decoder_passed = decoder
            if decoder == "fpn":
                # Infer the FPN level from the backbone. If user need to customize
                # this setting, they should manually create the FPN and backbone.
                if not isinstance(backbone.output, dict):
                    raise ValueError(
                        "Expect the backbone's output to be dict, "
                        f"received {backbone.output}"
                    )
                backbone_levels = list(backbone.output.keys())
                min_level = backbone_levels[0]
                max_level = backbone_levels[-1]
                decoder = keras_cv.layers.FeaturePyramid(
                    min_level=min_level, max_level=max_level
                )

        # TODO(scottzhu): do more validation for the decoder when we have a common
        # interface.
        self.decoder = decoder

        self._segmentation_head_passed = segmentation_head
        if segmentation_head is None:
            # Scale up the output when using FPN, to keep the output shape same as the
            # input shape.
            if isinstance(self.decoder, keras_cv.layers.FeaturePyramid):
                output_scale_factor = pow(2, self.decoder.min_level)
            else:
                output_scale_factor = None

            segmentation_head = (
                keras_cv.models.segmentation.__internal__.SegmentationHead(
                    classes=classes, output_scale_factor=output_scale_factor
                )
            )
        self.segmentation_head = segmentation_head

    def call(self, inputs, training=None):
        backbone_output = self.backbone(inputs, training=training)
        decoder_output = self.decoder(backbone_output, training=training)
        return self.segmentation_head(decoder_output, training=training)

    # TODO(tanzhenyu): consolidate how regularization should be applied to KerasCV.
    def compile(self, weight_decay=0.0001, **kwargs):
        self.weight_decay = weight_decay
        super().compile(**kwargs)

    def train_step(self, data):
        images, y_true, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(images, training=True)
            total_loss = self.compute_loss(images, y_true, y_pred)
            reg_losses = []
            if self.weight_decay:
                for var in self.trainable_variables:
                    if "bn" not in var.name:
                        reg_losses.append(0.0001 * tf.nn.l2_loss(var))
                l2_loss = tf.math.add_n(reg_losses)
                total_loss += l2_loss
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(images, y_true, y_pred, sample_weight=None)

    def get_config(self):
        config = {
            "classes": self.classes,
            "backbone": self._backbone_passed,
            "decoder": self._decoder_passed,
            "segmentation_head": self._segmentation_head_passed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
