import tensorflow as tf

from keras_cv.layers.preprocessing import Rescaling
from keras_cv.models import VGG16
from keras_cv.models import VGG19

BACKBONE_CONFIG = {
    "vgg16": {
        "BLOCK3": "block3_pool",
        "BLOCK4": "block4_pool",
        "BLOCK5": "block5_pool",
        "DENSE_START": "fc1",
        "DENSE_END": "predictions",
    },
    "vgg19": {
        "BLOCK3": "block3_pool",
        "BLOCK4": "block4_pool",
        "BLOCK5": "block5_pool",
        "DENSE_START": "fc1",
        "DENSE_END": "predictions",
    },
}
BACKBONE = {"vgg16": VGG16, "vgg19": VGG19}


def getPoolLayers(model, pool_id):
    """Function to get the all layers upto a certain named layer. Used only for extracting MaxPooling2D-based layers.

    Args:
        model: `tf.keras.layers.Model` or `tf.keras.Sequential`, The original model from which the layers are extracted
        pool_id: str, The name of the layer that acts as the end layer limit for the extraction
    Returns:
        layers: `tf.keras.Sequential` representing the extracted layers
    """
    layer_names = [layer.name for layer in model.layers]

    pool_layers = []
    for layer_id in layer_names:
        layer = model.get_layer(layer_id)
        pool_layers.append(layer)
        if layer_id == pool_id:
            return tf.keras.Sequential(pool_layers)


def getDenseToConvolutionLayers(model, dense_start_id, dense_end_id):
    """Utility function to convert all `tf.keras.layers.Dense` functions to `tf.keras.layers.Conv2D` with `filters=units` from the Dense layers and `kernel_size=(1,1)`, `padding='same'`

    Args:
        model: `tf.keras.layers.Model` or `tf.keras.Sequential`, The original model from which the layers are extracted
        dense_start_id: str, The name of the layer that acts as the start layer limit for the extraction
        dense_end_id: str, The name of the layer that acts as the end layer limit for the extraction
    Returns:
        layers: `tf.keras.Sequential` representing the converted layers.
    """
    layer_names = [layer.name for layer in model.layers]
    dense_flag = False

    units = []

    for layer_id in layer_names:
        if layer_id == dense_start_id or dense_flag == True:
            dense_flag = True
            layer = model.get_layer(layer_id)
            units.append(layer.units)
            if layer_id == dense_end_id:
                dense_flag = False
                break

    dense_convs = [
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            activation="relu",
            padding="same",
        )
        for filters in units
    ]
    return tf.keras.Sequential(dense_convs)


def VGGBackboneBuilder(
    model_version,
    model_architecture,
    include_rescaling,
    classes,
    input_shape=(224, 224, 3),
):
    """Utility function to build the backbone of the FCN with variants of the VGG model.

    Args:
        model_version: str, one of 'vgg16' or 'vgg19'. Defines which model to use as the backbone base.
        model_architecture: str, one of 'fcn8s', 'fcn16s' or 'fcn32s'. Defines which architecture and sampling method should be used as detailed in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
        include_rescaling: bool, one of True or False. Defines whether to use a Rescaling layer or not.
        classes: int. Defines the number of classes for the output.
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.

    Returns:
        model: tf.keras.models.Model. Represents the graph of all backbone operations and outputs for the chosen architecture.
    """
    vgg_model = BACKBONE[model_version](
        include_rescaling=include_rescaling,
        classes=classes,
        input_shape=input_shape,
        include_top=True,
    )

    x = tf.keras.Input(shape=input_shape)

    if model_architecture == "fcn8s":
        # Made it like this, because then parameter sharing occurs to get the model parameter size to go down drastically
        pool5 = getPoolLayers(vgg_model, BACKBONE_CONFIG[model_version]["BLOCK5"])

        pool3_output, pool4_output = x, x

        for layer in pool5.layers:
            pool3_output = layer(pool3_output)
            if layer.name == BACKBONE_CONFIG[model_version]["BLOCK3"]:
                break
        for layer in pool5.layers:
            pool4_output = layer(pool4_output)
            if layer.name == BACKBONE_CONFIG[model_version]["BLOCK4"]:
                break

        dense_convs = getDenseToConvolutionLayers(
            model=vgg_model,
            dense_start_id=BACKBONE_CONFIG[model_version]["DENSE_START"],
            dense_end_id=BACKBONE_CONFIG[model_version]["DENSE_END"],
        )
        pool5_output = pool5(x)
        pool5_output = dense_convs(pool5_output)

        model = tf.keras.models.Model(
            inputs=x,
            outputs={
                "pool3": pool3_output,
                "pool4": pool4_output,
                "pool5": pool5_output,
            },
        )

        return model
    elif model_architecture == "fcn16s":

        pool5 = getPoolLayers(vgg_model, BACKBONE_CONFIG[model_version]["BLOCK5"])

        pool4_output = x

        for layer in pool5.layers:
            pool4_output = layer(pool4_output)
            if layer.name == BACKBONE_CONFIG[model_version]["BLOCK4"]:
                break

        dense_convs = getDenseToConvolutionLayers(
            model=vgg_model,
            dense_start_id=BACKBONE_CONFIG[model_version]["DENSE_START"],
            dense_end_id=BACKBONE_CONFIG[model_version]["DENSE_END"],
        )
        pool5_output = pool5(x)
        pool5_output = dense_convs(pool5_output)

        model = tf.keras.models.Model(
            inputs=x, outputs={"pool4": pool4_output, "pool5": pool5_output}
        )

        return model
    elif model_architecture == "fcn32s":

        pool5 = getPoolLayers(vgg_model, BACKBONE_CONFIG[model_version]["BLOCK5"])

        dense_convs = getDenseToConvolutionLayers(
            model=vgg_model,
            dense_start_id=BACKBONE_CONFIG[model_version]["DENSE_START"],
            dense_end_id=BACKBONE_CONFIG[model_version]["DENSE_END"],
        )
        pool5_output = pool5(x)
        pool5_output = dense_convs(pool5_output)

        model = tf.keras.models.Model(inputs=x, outputs={"pool5": pool5_output})

        return model


def VGGArchitectureBuilder(
    model_version,
    model_architecture,
    include_rescaling,
    classes,
    input_tensor,
    input_shape=(224, 224, 3),
):
    """Main function for development and execution of full FCN model specifically for VGG-based backbones.

    Args:
        model_version: str, one of 'vgg16' or 'vgg19'. Defines which model to use as the backbone base.
        model_architecture: str, one of 'fcn8s', 'fcn16s' or 'fcn32s'. Defines which architecture and sampling method should be used as detailed in the paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
        include_rescaling: bool, one of True or False. Defines whether to use a Rescaling layer or not.
        classes: int. Defines the number of classes for the output.
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.
    Returns:
        model: `tf.Tensor` representing the output of the FCN model. Shape should match the format (input_shape[1], input_shape[2], classes)

    """
    if model_version not in ["vgg16", "vgg19"]:
        raise ValueError(
            "Chosen `backbone` argument is not a valid allowed backbone. Possible options are ['vgg16', 'vgg19']"
        )
    if model_architecture == "fcn8s":
        backbone = VGGBackboneBuilder(
            model_version=model_version,
            include_rescaling=include_rescaling,
            classes=classes,
            input_shape=input_shape,
            model_architecture="fcn8s",
        )
        backbone_output = backbone(input_tensor)
        pool3, pool4, pool5 = (
            backbone_output["pool3"],
            backbone_output["pool4"],
            backbone_output["pool5"],
        )

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format="channels_last", interpolation="bilinear"
        )
        pool5 = pool5_upsampling(pool5)

        pool3 = tf.keras.layers.Conv2D(
            filters=pool5.shape[-1], kernel_size=(1, 1), padding="same", strides=(1, 1)
        )(pool3)
        pool4 = tf.keras.layers.Conv2D(
            filters=pool5.shape[-1], kernel_size=(1, 1), padding="same", strides=(1, 1)
        )(pool4)

        intermediate_pool_output = tf.keras.layers.Add()([pool4, pool5])

        intermediate_pool_output = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format="channels_last", interpolation="bilinear"
        )(intermediate_pool_output)

        final_pool_output = tf.keras.layers.Add()([pool3, intermediate_pool_output])

        output_layer = tf.keras.layers.UpSampling2D(
            size=(8, 8), data_format="channels_last", interpolation="bilinear"
        )
        return output_layer(final_pool_output)

    elif model_architecture == "fcn16s":
        backbone = VGGBackboneBuilder(
            model_version=model_version,
            include_rescaling=include_rescaling,
            classes=classes,
            input_shape=input_shape,
            model_architecture="fcn16s",
        )
        backbone_output = backbone(input_tensor)
        pool4, pool5 = backbone_output["pool4"], backbone_output["pool5"]

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format="channels_last", interpolation="bilinear"
        )
        pool5 = pool5_upsampling(pool5)

        pool4 = tf.keras.layers.Conv2D(
            filters=pool5.shape[-1], kernel_size=(1, 1), padding="same", strides=(1, 1)
        )(pool4)

        final_pool_output = tf.keras.layers.Add()([pool4, pool5])

        output_layer = tf.keras.layers.UpSampling2D(
            size=(16, 16), data_format="channels_last", interpolation="bilinear"
        )
        return output_layer(final_pool_output)

    elif model_architecture == "fcn32s":
        backbone = VGGBackboneBuilder(
            model_version=model_version,
            include_rescaling=include_rescaling,
            classes=classes,
            input_shape=input_shape,
            model_architecture="fcn32s",
        )
        backbone_output = backbone(input_tensor)
        pool5 = backbone_output["pool5"]

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(32, 32), data_format="channels_last", interpolation="bilinear"
        )

        return pool5_upsampling(pool5)


def CustomArchitectureBuilder(model):
    """Utility function to parse a tf.keras.Model structure and get a FCN backbone from it. It maps all Conv2D and MaxPooling layers directly as a 1-to-1 port, while it converts all Dense layers into 1x1 Conv2D layers.

    Args:
        model: `tf.keras.models.Model` instance. Defines the custom backbone model passed as an argument.

    Returns:
        backbone: `tf.keras.Sequential` instance defining the FCN-specific parse output for the model passed as argument.

    Warning:
        It has undefined behaviour for ResNet-style, skip-connection based models.
    """
    layer_list = []
    dense_units_list = []
    # Design choice : Either fully reject, or pick up only Conv2D and MaxPooling2D layers and convert Dense to Conv in `build()`
    # Possible edge case : ResNets
    # Current design allows a simple CNN with Conv2D, MaxPooling2D and Dense layers only, to be parsed and made into a FCN directly.
    for i in model.layers:
        if isinstance(i, tf.keras.layers.Conv2D) or isinstance(
            i, tf.keras.layers.MaxPooling2D
        ):
            layer_list.append(i)
        elif isinstance(i, tf.keras.layers.Dense):
            dense_units_list.append(i.units)
            dense_convs = [
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                )
                for filters in dense_units_list
            ]
            layer_list.append(*dense_convs)
    if len(layer_list) == 0:
        raise ValueError(
            "Entered `backbone` argument does not have any Conv2D or MaxPooling layers. Include a `tf.keras.models.Model` with `keras.layers.Conv2D` or `keras.layers.MaxPooling2D` layers"
        )
    if not all(
        isinstance(layer, tf.keras.layers.Conv2D)
        or isinstance(layer, tf.keras.layers.MaxPooling2D)
        or isinstance(layer, tf.keras.layers.Dense)
        for layer in layer_list
    ):
        raise ValueError(
            "Entered `backbone` argument has custom layers. Include a `tf.keras.models.Model` with `tf.keras.layers.Conv2D`, `tf.keras.layers.Dense` or `tf.keras.layers.MaxPooling2D` layers only."
        )
    return tf.keras.Sequential(layer_list)


class FCN(tf.keras.models.Model):
    """A segmentation model based on the [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf) paper introduced by Long et. al.

    Args:
        classes: int. Defines the number of classes for the output.
        backbone: One of 'vgg16', 'vgg19' or a `tf.keras.models.Model` instance. Used as the main backbone for the complete FCN.
            If a custom `tf.keras.models.Model` instance is passed, `model_architecture` cannot be passed. A custom model can only contain
            `tf.keras.layers.Conv2D`, `tf.keras.layers.MaxPooling2D` or `tf.keras.layers.Dense` only.
        model_architecture: One of 'fcn8s', 'fcn16s' or 'fcn32s'. Used to define the up-sampling scale and architecture to be used for the FCN. This option is only available for pre-supported backbones ('VGG16' and 'VGG19').
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.
        include_rescaling: bool, one of True or False. Defines whether to use a Rescaling layer or not.
        return_mask: bool, one of True or False. Returns a 1-channel result instead of a num_classes-channel result.
    """

    def __init__(
        self,
        classes,
        backbone,
        model_architecture=None,
        input_shape=(224, 224, 3),
        include_rescaling=False,
        return_mask=False,
    ):

        if isinstance(backbone, tf.keras.models.Model):
            if model_architecture is not None:
                raise ValueError(
                    "`model_architecture` cannot be set if `backbone` is not in ['vgg16', 'vgg19']. Either set `backbone` to one of the accepted values or remove the `model_architecture` argument."
                )
            else:
                self.backbone = CustomArchitectureBuilder(backbone)
                self.classes_conv = tf.keras.layers.Conv2D(
                    filters=classes, kernel_size=(1, 1), strides=(1, 1), padding="same"
                )

                output_shape = self.backbone.layers[-1].compute_output_shape()

                target_height_factor = self.height // output_shape[1]
                target_width_factor = self.width // output_shape[2]

                self.upscale = tf.keras.layers.UpSampling2D(
                    size=(target_height_factor, target_width_factor),
                    data_format="channels_last",
                    interpolation="bilinear",
                )

                x = tf.keras.Input(input_shape)
                if include_rescaling:
                    x = Rescaling(scale=1.0 / 255)(x)
                x = self.backbone(x)
                x = self.classes_conv(x)
                output_tensor = self.upscale(x)
                if return_mask:
                    # Assumes channels_last
                    output_tensor = tf.math.argmax(output_tensor, axis=3)
                    output_tensor = tf.expand_dims(output_tensor, axis=3)

                super().__init__(
                    inputs={"input_tensor": x}, outputs={"output_tensor": output_tensor}
                )
                self.classes = classes
                self.model_architecture = "custom"
                self.input_shape = input_shape
                self.include_rescaling = include_rescaling
        elif isinstance(backbone, str):
            if model_architecture not in ["fcn8s", "fcn16s", "fcn32s"]:
                raise ValueError(
                    "Invalid argument for parameter `model_architecture`. Accepted values are ['fcn8s', 'fcn16s', 'fcn32s']"
                )
            else:
                input_tensor = tf.keras.Input(shape=input_shape)
                output_tensor = VGGArchitectureBuilder(
                    classes=classes,
                    model_version=backbone,
                    model_architecture=model_architecture,
                    include_rescaling=include_rescaling,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                )
                if return_mask:
                    # Assumes channels_last
                    output_tensor = tf.math.argmax(output_tensor, axis=3)
                    output_tensor = tf.expand_dims(output_tensor, axis=3)

                super().__init__(
                    inputs={"input_tensor": input_tensor},
                    outputs={"output_tensor": output_tensor},
                )
        else:
            raise ValueError(
                "Invalid argument for parameter `backbone`. Accepted values are ['vgg16', 'vgg19'] or a `tf.keras.models.Model` instance with only `tf.keras.layers.Conv2D`, 'tf.keras.layers.MaxPooling2D' or `tf.keras.layers.Dense` layers"
            )


def FCN8S(classes, input_shape, include_rescaling, backbone):
    if not isinstance(backbone, str) and backbone not in ["vgg16", "vgg19"]:
        raise ValueError(
            "Invalid argument for parameter `backbone`. Accepted values are ['vgg16', 'vgg19']"
        )
    return FCN(
        classes=classes,
        input_shape=input_shape,
        include_rescaling=include_rescaling,
        backbone=backbone,
        model_architecture="fcn8s",
    )


def FCN16S(classes, input_shape, include_rescaling, backbone):
    if not isinstance(backbone, str) and backbone not in ["vgg16", "vgg19"]:
        raise ValueError(
            "Invalid argument for parameter `backbone`. Accepted values are ['vgg16', 'vgg19']"
        )
    return FCN(
        classes=classes,
        input_shape=input_shape,
        include_rescaling=include_rescaling,
        backbone=backbone,
        model_architecture="fcn16s",
    )


def FCN32S(classes, input_shape, include_rescaling, backbone):
    if not isinstance(backbone, str) and backbone not in ["vgg16", "vgg19"]:
        raise ValueError(
            "Invalid argument for parameter `backbone`. Accepted values are ['vgg16', 'vgg19']"
        )
    return FCN(
        classes=classes,
        input_shape=input_shape,
        include_rescaling=include_rescaling,
        backbone=backbone,
        model_architecture="fcn32s",
    )
