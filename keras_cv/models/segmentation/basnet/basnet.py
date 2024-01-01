import copy

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.models import utils
from keras_cv.models.backbones.backbone_presets import backbone_presets
from keras_cv.models.backbones.backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    apply_basic_block as resnet_basic_block
)
from keras_cv.models.segmentation.basnet.basnet_presets import (  # noqa: E501
    basnet_presets,
)
from keras_cv.models.task import Task
from keras_cv.utils.python_utils import classproperty


@keras_cv_export(
    [
        "keras_cv.models.BASNet",
        "keras_cv.models.segmentation.BASNet",
    ]
)
class BASNET(Task):

    def __init__(
            self,
            backbone,
            num_classes,
            input_shape=(None, None, 3),
            input_tensor=None,
            include_rescaling=False,
            projection_filters=64,
            prediction_head=None,
            refinement_head=None,
            **kwargs,
    ):
        if not isinstance(backbone, keras.layers.Layer) or not isinstance(
                backbone, keras.Model
        ):
            raise ValueError(
                "Argument `backbone` must be a `keras.layers.Layer` instance "
                f" or `keras.Model`. Received instead "
                f"backbone={backbone} (of type {type(backbone)})."
            )

        if backbone.input_shape != (None, None, None, 3):
            raise ValueError(
                "Do not specify 'input_shape' or 'input_tensor' within the 'BASNet' backbone."
                "\nPlease provide 'input_shape' or 'input_tensor' while initializing the 'BASNet' model."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        if prediction_head is None:
            input_size = keras.backend.int_shape(x)
            prediction_head = keras.Sequential([
                keras.layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same"),
                keras.layers.Resizing(input_size[1], input_size[2])
            ])
        if refinement_head is None:
            refinement_head = keras.Sequential([
                keras.layers.Conv2D(num_classes, kernel_size=(3, 3), padding="same"),
            ])

        # Prediction model.
        predict_model = basnet_predict(x, backbone, projection_filters, prediction_head)

        # Refinement model.
        refine_model = basnet_rrm(predict_model, projection_filters, refinement_head)

        outputs = [refine_model.output]  # Combine outputs.
        outputs.extend(predict_model.output)

        outputs = [keras.layers.Activation("sigmoid", dtype="float32")(_) for _ in outputs]  # Activations.

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling
        self.projection_filters = projection_filters
        self.prediction_head = prediction_head
        self.refinement_head = refinement_head

    def get_config(self):
        return {
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "input_shape": self.input_shape[1:],
            "input_tensor": keras.saving.serialize_keras_object(self.input_tensor),
            "include_rescaling": self.include_rescaling,
            "projection_filters": self.projection_filters,
            "prediction_head": keras.saving.serialize_keras_object(
                self.prediction_head
            ),
            "refinement_head": keras.saving.serialize_keras_object(
                self.refinement_head
            ),
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            if config["backbone"]["config"]["input_shape"] != (None, None, 3):
                config["input_shape"] = config["backbone"]["config"]["input_shape"]
                config["backbone"]["config"]["input_shape"] = (None, None, 3)
            config["backbone"] = keras.layers.deserialize(config["backbone"])

        if "input_tensor" in config and isinstance(
                config["input_tensor"], dict
        ):
            config["input_tensor"] = keras.layers.deserialize(
                config["input_tensor"]
            )

        if "prediction_head" in config and isinstance(
                config["prediction_head"], dict
        ):
            config["prediction_head"] = keras.layers.deserialize(
                config["prediction_head"]
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
        return copy.deepcopy({**backbone_presets, **basnet_presets})

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(
            {**backbone_presets_with_weights, **basnet_presets}
        )

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configurations of compatible
        backbones."""
        return copy.deepcopy(backbone_presets)


def convolution_block(x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = keras.layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def get_resnet_block(_resnet, block_num):
    """Extract and return ResNet-34 block."""
    extractor_levels = ["P2", "P3", "P4", "P5"]
    return keras.models.Model(
        inputs=_resnet.get_layer(f"v2_stack_{block_num}_block1_1_conv").input,
        outputs=_resnet.get_layer(
            _resnet.pyramid_level_inputs[extractor_levels[block_num]]
        ).output,
        name=f"resnet34_block{block_num + 1}",
    )


def basnet_predict(x_input, backbone, filters, segmentation_head):
    """BASNet Prediction Module, it outputs coarse label map."""
    num_stages = 6

    x = x_input

    # -------------Encoder--------------
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = get_resnet_block(backbone, i)(x)
            encoder_blocks.append(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            shape = keras.backend.int_shape(x)
            x = resnet_basic_block(x, filters=shape[3], conv_shortcut=False,
                                   name=f"v1_basic_block_{i + 1}_1")
            x = resnet_basic_block(x, filters=shape[3], conv_shortcut=False,
                                   name=f"v1_basic_block_{i + 1}_2")
            x = resnet_basic_block(x, filters=shape[3], conv_shortcut=False,
                                   name=f"v1_basic_block_{i + 1}_3")
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
            shape = keras.backend.int_shape(x)
            x = keras.layers.Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block)
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


def basnet_rrm(base_model, filters, segmentation_head):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 4

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = keras.backend.int_shape(x)
        x = keras.layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = keras.layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = keras.layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=[base_model.input], outputs=[x])
