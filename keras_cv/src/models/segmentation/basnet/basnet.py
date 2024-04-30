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

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone_presets import backbone_presets
from keras_cv.src.models.backbones.resnet_v1.resnet_v1_backbone import (
    apply_basic_block as resnet_basic_block,
)
from keras_cv.src.models.segmentation.basnet.basnet_presets import (
    basnet_presets,
)
from keras_cv.src.models.segmentation.basnet.basnet_presets import (
    presets_no_weights,
)
from keras_cv.src.models.segmentation.basnet.basnet_presets import (
    presets_with_weights,
)
from keras_cv.src.models.task import Task
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export(
    [
        "keras_cv.models.BASNet",
        "keras_cv.models.segmentation.BASNet",
    ]
)
class BASNet(Task):
    """
    A Keras model implementing the BASNet architecture for semantic
    segmentation.

    References:
        - [BASNet: Boundary-Aware Segmentation Network for Mobile and Web Applications](https://arxiv.org/abs/2101.04704)

    Args:
        backbone: `keras.Model`. The backbone network for the model that is
            used as a feature extractor for BASNet prediction encoder. Currently
            supported backbones are ResNet18 and ResNet34. Default backbone is
            `keras_cv.models.ResNet34Backbone()`
            (Note: Do not specify 'input_shape', 'input_tensor', or 'include_rescaling'
            within the backbone. Please provide these while initializing the
            'BASNet' model.)
        num_classes: int, the number of classes for the segmentation model.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.
        include_rescaling: bool, whether to rescale the inputs. If set
            to `True`, inputs will be passed through a `Rescaling(1/255.0)`
            layer.
        projection_filters: int, number of filters in the convolution layer
            projecting low-level features from the `backbone`.
        prediction_heads: (Optional) List of `keras.layers.Layer` defining
            the prediction module head for the model. If not provided, a
            default head is created with a Conv2D layer followed by resizing.
        refinement_head: (Optional) a `keras.layers.Layer` defining the
            refinement module head for the model. If not provided, a default
            head is created with a Conv2D layer.

    Example:
    ```python

    import keras_cv

    images = np.ones(shape=(1, 288, 288, 3))
    labels = np.zeros(shape=(1, 288, 288, 1))

    # Note: Do not specify 'input_shape', 'input_tensor', or
    # 'include_rescaling' within the backbone.
    backbone = keras_cv.models.ResNet34Backbone()
    model = keras_cv.models.segmentation.BASNet(
        backbone=backbone,
        num_classes=1,
        input_shape=[288, 288, 3],
        include_rescaling=False
    )

    # Evaluate model
    output = model(images)
    pred_labels = output[0]

    # Train model
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(images, labels, epochs=3)
        ```
    """  # noqa: E501

    def __init__(
        self,
        backbone,
        num_classes,
        input_shape=(None, None, 3),
        input_tensor=None,
        include_rescaling=False,
        projection_filters=64,
        prediction_heads=None,
        refinement_head=None,
        **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
            backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance"
                f" or `keras.Model`. Received instead"
                f" backbone={backbone} (of type {type(backbone)})."
            )

        if backbone.input_shape != (None, None, None, 3):
            raise ValueError(
                "Do not specify 'input_shape' or 'input_tensor' within the"
                " 'BASNet' backbone. \nPlease provide 'input_shape' or"
                " 'input_tensor' while initializing the 'BASNet' model."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        if prediction_heads is None:
            prediction_heads = []
            for size in (1, 2, 4, 8, 16, 32, 32):
                head_layers = [
                    keras.layers.Conv2D(
                        num_classes, kernel_size=(3, 3), padding="same"
                    )
                ]
                if size != 1:
                    head_layers.append(
                        keras.layers.UpSampling2D(
                            size=size, interpolation="bilinear"
                        )
                    )
                prediction_heads.append(keras.Sequential(head_layers))

        if refinement_head is None:
            refinement_head = keras.Sequential(
                [
                    keras.layers.Conv2D(
                        num_classes, kernel_size=(3, 3), padding="same"
                    ),
                ]
            )

        # Prediction model.
        predict_model = basnet_predict(
            x, backbone, projection_filters, prediction_heads
        )

        # Refinement model.
        refine_model = basnet_rrm(
            predict_model, projection_filters, refinement_head
        )

        outputs = refine_model.outputs  # Combine outputs.
        outputs.extend(predict_model.outputs)

        outputs = [
            keras.layers.Activation("sigmoid", dtype="float32")(_)
            for _ in outputs
        ]  # Activations.

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling
        self.projection_filters = projection_filters
        self.prediction_heads = prediction_heads
        self.refinement_head = refinement_head

    def get_config(self):
        return {
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "input_shape": self.input_shape[1:],
            "input_tensor": keras.saving.serialize_keras_object(
                self.input_tensor
            ),
            "include_rescaling": self.include_rescaling,
            "projection_filters": self.projection_filters,
            "prediction_heads": [
                keras.saving.serialize_keras_object(prediction_head)
                for prediction_head in self.prediction_heads
            ],
            "refinement_head": keras.saving.serialize_keras_object(
                self.refinement_head
            ),
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            input_shape = (None, None, 3)
            if isinstance(config["backbone"]["config"]["input_shape"], list):
                input_shape = list(input_shape)
            if config["backbone"]["config"]["input_shape"] != input_shape:
                config["input_shape"] = config["backbone"]["config"][
                    "input_shape"
                ]
                config["backbone"]["config"]["input_shape"] = input_shape
            config["backbone"] = keras.layers.deserialize(config["backbone"])

        if "input_tensor" in config and isinstance(
            config["input_tensor"], dict
        ):
            config["input_tensor"] = keras.layers.deserialize(
                config["input_tensor"]
            )

        if "prediction_heads" in config and isinstance(
            config["prediction_heads"], list
        ):
            for i in range(len(config["prediction_heads"])):
                if isinstance(config["prediction_heads"][i], dict):
                    config["prediction_heads"][i] = keras.layers.deserialize(
                        config["prediction_heads"][i]
                    )

        if "refinement_head" in config and isinstance(
            config["refinement_head"], dict
        ):
            config["refinement_head"] = keras.layers.deserialize(
                config["refinement_head"]
            )
        return super().from_config(config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        filtered_backbone_presets = copy.deepcopy(
            {
                k: v
                for k, v in backbone_presets.items()
                if k in ("resnet18", "resnet34")
            }
        )

        return copy.deepcopy({**filtered_backbone_presets, **basnet_presets})

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return copy.deepcopy(presets_with_weights)

    @classproperty
    def presets_without_weights(cls):
        """
        Dictionary of preset names and configurations that has no weights.
        """
        return copy.deepcopy(presets_no_weights)

    @classproperty
    def backbone_presets(cls):
        """
        Dictionary of preset names and configurations of compatible backbones.
        """
        filtered_backbone_presets = copy.deepcopy(
            {
                k: v
                for k, v in backbone_presets.items()
                if k in ("resnet18", "resnet34")
            }
        )
        filtered_presets = copy.deepcopy(filtered_backbone_presets)
        return filtered_presets


def convolution_block(x_input, filters, dilation=1):
    """
    Apply convolution + batch normalization + ReLU activation.

    Args:
        x_input: Input keras tensor.
        filters: int, number of output filters in the convolution.
        dilation: int, dilation rate for the convolution operation.
            Defaults to 1.

    Returns:
        A tensor with convolution, batch normalization, and ReLU
        activation applied.
    """
    x = keras.layers.Conv2D(
        filters, (3, 3), padding="same", dilation_rate=dilation
    )(x_input)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def get_resnet_block(_resnet, block_num):
    """
    Extract and return a specific ResNet block.

    Args:
        _resnet: `keras.Model`. ResNet model instance.
        block_num: int, block number to extract.

    Returns:
        A Keras Model representing the specified ResNet block.
    """

    extractor_levels = ["P2", "P3", "P4", "P5"]
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            _resnet.pyramid_level_inputs[extractor_levels[block_num]]
        ).output,
        name=f"resnet_block{block_num + 1}",
    )


def basnet_predict(x_input, backbone, filters, segmentation_heads):
    """
    BASNet Prediction Module.

    This module outputs a coarse label map by integrating heavy
    encoder, bridge, and decoder blocks.

    Args:
        x_input: Input keras tensor.
        backbone: `keras.Model`. The backbone network used as a feature
            extractor for BASNet prediction encoder.
        filters: int, the number of filters.
        segmentation_heads: List of `keras.layers.Layer`, A list of Keras
            layers serving as the segmentation head for prediction module.


    Returns:
        A Keras Model that integrates the encoder, bridge, and decoder
        blocks for coarse label map prediction.
    """
    num_stages = 6

    x = x_input

    # -------------Encoder--------------
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet backbone.
            x = get_resnet_block(backbone, i)(x)
            encoder_blocks.append(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            for j in range(3):
                x = resnet_basic_block(
                    x,
                    filters=x.shape[3],
                    conv_shortcut=False,
                    name=f"v1_basic_block_{i + 1}_{j + 1}",
                )
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)

        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block)  # Prediction segmentation head.
        for segmentation_head, decoder_block in zip(
            segmentation_heads, decoder_blocks
        )
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


def basnet_rrm(base_model, filters, segmentation_head):
    """
    BASNet Residual Refinement Module (RRM).

    This module outputs a fine label map by integrating light encoder,
    bridge, and decoder blocks.

    Args:
        base_model: Keras model used as the base or coarse label map.
        filters: int, the number of filters.
        segmentation_head: a `keras.layers.Layer`, A Keras layer serving
            as the segmentation head for refinement module.

    Returns:
        A Keras Model that constructs the Residual Refinement Module (RRM).
    """
    num_stages = 4

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(
        x_input
    )

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        x = keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x)  # Refinement segmentation head.

    # ------------- refined = coarse + residual
    x = keras.layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=base_model.input, outputs=[x])
