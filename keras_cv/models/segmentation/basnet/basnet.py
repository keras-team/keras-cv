from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.models import utils
from keras_cv.models.backbones.resnet_v1.resnet_v1_backbone import (
    apply_basic_block as resnet_basic_block
)
from keras_cv.models.task import Task


@keras_cv_export(
    [
        "keras_cv.models.BASNET",
        "keras_cv.models.segmentation.BASNET",
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
            segmentation_head=None,
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
                "Do not specify 'input_shape' or 'input_tensor' within the 'BASNet' backbone. "
                "Please provide 'input_shape' or 'input_tensor' while initializing the 'BASNet' model."
            )

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(1 / 255.0)(x)

        # Prediction model.
        predict_model = basnet_predict(x, backbone, projection_filters, num_classes)

        # Refinement model.
        refine_model = basnet_rrm(predict_model, projection_filters, num_classes)

        outputs = [refine_model.output]  # Combine outputs.
        outputs.extend(predict_model.output)

        outputs = [keras.layers.Activation("sigmoid")(_) for _ in outputs]  # Activations.

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

        self.backbone = backbone
        self.num_classes = num_classes
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling
        self.projection_filters = projection_filters
        self.segmentation_head = segmentation_head


def convolution_block(x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = keras.layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = keras.layers.BatchNormalization()(x)
    return keras.layers.Activation("relu")(x)


def segmentation_head(x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = keras.layers.Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = keras.layers.Resizing(final_size[0], final_size[1])(x)

    return x


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


def basnet_predict(x_input, backbone, filters, num_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    num_stages = 6

    x = x_input
    input_shape = keras.backend.int_shape(x)

    # -------------Encoder--------------
    x = keras.layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x)

    # backbone = backbone(
    #     include_rescaling=False,# input_shape=[288, 288, 3]
    # )

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
        segmentation_head(decoder_block, num_classes, input_shape[1:3])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=[x_input], outputs=decoder_blocks)


def basnet_rrm(base_model, filters, out_classes):
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

    x = segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = keras.layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=[base_model.input], outputs=[x])
