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

from keras_cv.models import VGG16, VGG19
from keras_cv.models import utils

BACKBONE_CONFIG = {
    "vgg16": {
        "BLOCK3": "block3_pool",
        "BLOCK4": "block4_pool",
        "BLOCK5": "block5_pool",
        "DENSE_UNITS": [4096, 4096],
    },
    "vgg19": {
        "BLOCK3": "block3_pool",
        "BLOCK4": "block4_pool",
        "BLOCK5": "block5_pool",
        "DENSE_UNITS": [4096, 4096],
    },
}
BACKBONE = {"vgg16": VGG16, "vgg19": VGG19}


def get_dense_to_convolution_layers(model):
    """Utility function to convert all `tf.keras.layers.Dense` functions to `tf.keras.layers.Conv2D` with `filters=units` from the Dense layers and `kernel_size=(1,1)`, `padding='same'`

    Args:
        model: `tf.keras.layers.Model` or `tf.keras.Sequential`, The original model from which the layers are extracted
    Returns:
        layers: `tf.keras.Sequential` representing the converted layers.
    """

    if model.name.lower() == "vgg16":
        backbone_name = "vgg16"
    elif model.name.lower() == "vgg19":
        backbone_name = "vgg19"

    units = BACKBONE_CONFIG[backbone_name]["DENSE_UNITS"]
    dense_convs = []

    for filter_idx in range(len(units)):
        filter = units[filter_idx]
        if filter_idx == 0:
            dense_layer = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=(7, 7,),
                strides=(1, 1),
                activation="relu",
                padding="same",
            )
            dense_convs.append(dense_layer)
        else:
            dense_layer = tf.keras.layers.Conv2D(
                filters=filter,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation="relu",
                padding="same",
            )
            dense_convs.append(dense_layer)
        dropout_layer = tf.keras.layers.Dropout(0.5)
        dense_convs.append(dropout_layer)
    return tf.keras.Sequential(dense_convs)


def vgg_backbone_builder(
    backbone, model_architecture, input_shape=(None, None, 3), input_tensor=None
):
    """Utility function to build the backbone of the FCN with variants of the VGG model.

    Args:
        backbone_name: str, one of 'vgg16' or 'vgg19'. Defines which model to use as the backbone base.
        model_architecture: str, one of 'fcn8s', 'fcn16s' or 'fcn32s'.
            Defines which architecture and sampling method should be used as detailed in the
            paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.

    Returns:
        model: tf.keras.models.Model. Represents the graph of all backbone operations and outputs for the chosen architecture.
    """
    vgg_model = backbone
    if vgg_model.name.lower() == "vgg16":
        backbone_name = "vgg16"
    else:
        backbone_name = "vgg19"

    x = utils.parse_model_inputs(input_shape=input_shape, input_tensor=input_tensor)

    if model_architecture == "fcn8s":
        # Made it like this, because then parameter sharing occurs to get the model parameter size to go down drastically
        pool5 = tf.keras.Model(
            inputs=vgg_model.layers[0].input,
            outputs=vgg_model.get_layer(
                BACKBONE_CONFIG[backbone_name]["BLOCK5"]
            ).output,
        )

        pool3_output, pool4_output = x, x

        for layer in pool5.layers:
            pool3_output = layer(pool3_output)
            if layer.name == BACKBONE_CONFIG[backbone_name]["BLOCK3"]:
                break
        for layer in pool5.layers:
            pool4_output = layer(pool4_output)
            if layer.name == BACKBONE_CONFIG[backbone_name]["BLOCK4"]:
                break

        dense_convs = get_dense_to_convolution_layers(model=vgg_model)
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

        pool5 = tf.keras.Model(
            inputs=vgg_model.layers[0].input,
            outputs=vgg_model.get_layer(
                BACKBONE_CONFIG[backbone_name]["BLOCK5"]
            ).output,
        )

        pool4_output = x

        for layer in pool5.layers:
            pool4_output = layer(pool4_output)
            if layer.name == BACKBONE_CONFIG[backbone_name]["BLOCK4"]:
                break

        dense_convs = get_dense_to_convolution_layers(model=vgg_model)
        pool5_output = pool5(x)
        pool5_output = dense_convs(pool5_output)

        model = tf.keras.models.Model(
            inputs=x, outputs={"pool4": pool4_output, "pool5": pool5_output}
        )

        return model
    elif model_architecture == "fcn32s":

        pool5 = tf.keras.Model(
            inputs=vgg_model.layers[0].input,
            outputs=vgg_model.get_layer(
                BACKBONE_CONFIG[backbone_name]["BLOCK5"]
            ).output,
        )

        dense_convs = get_dense_to_convolution_layers(model=vgg_model)
        pool5_output = pool5(x)
        pool5_output = dense_convs(pool5_output)

        model = tf.keras.models.Model(inputs=x, outputs={"pool5": pool5_output})

        return model


def vgg_architecture_builder(
    backbone, model_architecture, classes, input_tensor, input_shape=None,
):
    """Main function for development and execution of full FCN model specifically for VGG-based backbones.

    Args:
        backbone: str, one of 'vgg16' or 'vgg19'. Defines which model to use as the backbone base.
        model_architecture: str, one of 'fcn8s', 'fcn16s' or 'fcn32s'.
            Defines which architecture and sampling method should be used as detailed in the
            paper [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)
        classes: int. Defines the number of classes for the output.
        input_tensor: KerasTensor object. Defines the input `KerasTensor`.
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.
    Returns:
        model: `tf.Tensor` representing the output of the FCN model. Shape should match the
        format (input_shape[1], input_shape[2], classes)

    """
    if model_architecture == "fcn8s":
        backbone = vgg_backbone_builder(
            backbone=backbone,
            input_shape=input_shape,
            model_architecture="fcn8s",
            input_tensor=input_tensor,
        )
        backbone_output = backbone(input_tensor)
        pool3, pool4, pool5 = (
            backbone_output["pool3"],
            backbone_output["pool4"],
            backbone_output["pool5"],
        )

        pool3 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool3)

        pool4 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool4)

        pool5 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool5)

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )
        pool5 = pool5_upsampling(pool5)

        intermediate_pool_output = tf.keras.layers.Add()([pool4, pool5])

        intermediate_pool_output = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )(intermediate_pool_output)

        final_pool_output = tf.keras.layers.Add()([pool3, intermediate_pool_output])

        output_conv_layer = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            activation="softmax",
            padding="same",
            strides=(1, 1),
        )

        output_upsample_layer = tf.keras.layers.UpSampling2D(
            size=(8, 8),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )

        final_output = output_conv_layer(final_pool_output)
        final_output = output_upsample_layer(final_output)

        return final_output

    elif model_architecture == "fcn16s":
        backbone = vgg_backbone_builder(
            backbone=backbone,
            input_shape=input_shape,
            model_architecture="fcn16s",
            input_tensor=input_tensor,
        )
        backbone_output = backbone(input_tensor)
        pool4, pool5 = backbone_output["pool4"], backbone_output["pool5"]

        pool4 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool4)

        pool5 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool5)

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )
        pool5 = pool5_upsampling(pool5)

        final_pool_output = tf.keras.layers.Add()([pool4, pool5])

        output_conv_layer = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            activation="softmax",
            padding="same",
            strides=(1, 1),
        )

        output_upsample_layer = tf.keras.layers.UpSampling2D(
            size=(16, 16),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )

        final_output = output_conv_layer(final_pool_output)
        final_output = output_upsample_layer(final_output)

        return final_output

    elif model_architecture == "fcn32s":
        backbone = vgg_backbone_builder(
            backbone=backbone,
            input_shape=input_shape,
            model_architecture="fcn32s",
            input_tensor=input_tensor,
        )
        backbone_output = backbone(input_tensor)
        pool5 = backbone_output["pool5"]

        pool5 = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            padding="same",
            strides=(1, 1),
            activation="relu",
        )(pool5)

        pool5_upsampling = tf.keras.layers.UpSampling2D(
            size=(32, 32),
            data_format=tf.keras.backend.image_data_format(),
            interpolation="bilinear",
        )

        output_conv_layer = tf.keras.layers.Conv2D(
            filters=classes,
            kernel_size=(1, 1),
            activation="softmax",
            padding="same",
            strides=(1, 1),
        )

        final_output = output_conv_layer(pool5)
        final_output = pool5_upsampling(final_output)

        return final_output


def custom_architecture_builder(model):
    """Utility function to parse a tf.keras.Model structure and get a FCN backbone from it.
        It maps all layers directly as a 1-to-1 port, while it
        converts all Dense layers into 1x1 Conv2D layers.

    Args:
        model: `tf.keras.models.Model` instance. Defines the custom backbone model passed as an argument.

    Returns:
        backbone: `tf.keras.Sequential` instance defining the FCN-specific parse output for the model passed as argument.

    Warning:
        It has undefined behaviour for ResNet-style, skip-connection based models.
    """
    layer_list = []
    # Design choice : Either fully reject, or pick up only Conv2D and MaxPooling2D layers
    # and convert Dense to Conv in `build()`
    # Possible edge case : ResNets
    # Current design allows a simple sequential model only,
    # to be parsed and made into a FCN directly.
    for i in model.layers:
        if not isinstance(i, tf.keras.layers.Dense):
            layer_list.append(i)
        elif isinstance(i, tf.keras.layers.Dense):
            filters = i.units
            layer = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                activation="relu",
                padding="same",
            )
            layer_list.append(layer)
    if len(layer_list) == 0:
        raise ValueError(
            "Entered `backbone` argument does not have any layers. Include a `tf.keras.models.Model` with `keras.layers.*` layers"
        )
    return tf.keras.Sequential(layer_list)


class FullyConvolutionalNetwork(tf.keras.models.Model):
    """A segmentation model based on the [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf) paper introduced by Long et. al.

    Args:
        classes: int, the number of classes for the segmentation model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        backbone: a backbone network for the model. Can be a `tf.keras.models.Model`
            instance. The supported pre-defined backbone models are:
            1. A `keras_cv.models.VGG16` model
            2. A `keras_cv.models.VGG19` model
            Note: If a custom `tf.keras.models.Model` is passed, then only the
            `Conv2D`, `MaxPooling2D` and `Dense` layers are extracted from it
            to make the custom backbone.
        model_architecture: str, defines the model architecture based on the
            implementation details present in the paper. The supported
            architectures are:
            1. 'fcn8s', a FCN-8S definition
            2. 'fcn16s', a FCN-16S definition
            3. 'fcn32s', a FCN-32S definition
        input_shape: `list` or `tuple`. Defines the shape of the tensor to be expected as input.
        input_tensor: KerasTensor object. Defines the input `KerasTensor`.
        return_mask: bool, one of True or False. Returns a 1-channel result instead of a
            num_classes-channel result.
    """

    def __init__(
        self,
        classes,
        backbone,
        model_architecture=None,
        input_shape=None,
        input_tensor=None,
        return_mask=False,
    ):
        if backbone.name.lower() == "vgg16" or backbone.name.lower() == "vgg19":
            # Cannot run `isinstance` check as VGG16 and VGG19 are only functions and not classes.
            if model_architecture not in ["fcn8s", "fcn16s", "fcn32s"]:
                raise ValueError(
                    "Invalid argument for parameter `model_architecture`. Accepted values are ['fcn8s', 'fcn16s', 'fcn32s']"
                )
            else:
                if len(input_shape) != 4:
                    input_tensor = utils.parse_model_inputs(input_shape, input_tensor)
                else:
                    input_tensor = tf.keras.layers.Input(
                        shape=(input_shape[1], input_shape[2], input_shape[3])
                    )
                output_tensor = vgg_architecture_builder(
                    classes=classes,
                    backbone=backbone,
                    model_architecture=model_architecture,
                    input_tensor=input_tensor,
                    input_shape=input_shape,
                )

                data_format = tf.keras.backend.image_data_format()
                if return_mask:
                    if data_format == "channels_last":
                        output_tensor = tf.math.argmax(output_tensor, axis=3)
                        output_tensor = tf.expand_dims(output_tensor, axis=3)
                    else:
                        output_tensor = tf.math.argmax(output_tensor, axis=1)
                        output_tensor = tf.expand_dims(output_tensor, axis=1)

                output_tensor = tf.cast(output_tensor, tf.dtypes.float32)

                super().__init__(
                    inputs={"input_tensor": input_tensor},
                    outputs={"output_tensor": output_tensor},
                )

                self.classes = classes
                self.model_architecture = model_architecture
                self.backbone = backbone
                self.return_mask = return_mask

        elif isinstance(backbone, tf.keras.models.Model) or isinstance(
            backbone, tf.keras.Functional
        ):
            if model_architecture is not None:
                raise ValueError(
                    "`model_architecture` cannot be set if `backbone` is not a `keras_cv.models.VGG16` or `keras_cv.models.VGG19`. Either set `backbone` to one of the accepted values or remove the `model_architecture` argument."
                )
            else:
                if len(input_shape) != 4:
                    input_tensor = utils.parse_model_inputs(input_shape, input_tensor)
                else:
                    input_tensor = tf.keras.layers.Input(
                        shape=(input_shape[1], input_shape[2], input_shape[3])
                    )
                backbone = custom_architecture_builder(backbone)
                classes_conv = tf.keras.layers.Conv2D(
                    filters=classes,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="same",
                    activation="softmax",
                )

                input_shape = input_tensor.shape

                x = backbone(input_tensor)

                output_shape = x.shape

                if input_shape[0] is None and input_shape[1] is None:
                    if None in backbone.input.shape:
                        raise ValueError(
                            f"No `input_shape` argument available to model. `input_shape` received by `FullyConvolutionalNetwork` is {input_shape} and `input_shape` passed to backbone is {backbone.input.shape}"
                        )
                    else:
                        input_shape = backbone.input.shape

                target_height_factor = input_shape[1] // output_shape[1]
                target_width_factor = input_shape[2] // output_shape[2]

                data_format = tf.keras.backend.image_data_format()

                upscale = tf.keras.layers.UpSampling2D(
                    size=(target_height_factor, target_width_factor),
                    data_format=data_format,
                    interpolation="bilinear",
                )

                x = classes_conv(x)
                output_tensor = upscale(x)

                if return_mask:
                    if data_format == "channels_last":
                        output_tensor = tf.math.argmax(output_tensor, axis=3)
                        output_tensor = tf.expand_dims(output_tensor, axis=3)
                    else:
                        output_tensor = tf.math.argmax(output_tensor, axis=1)
                        output_tensor = tf.expand_dims(output_tensor, axis=1)

                output_tensor = tf.cast(output_tensor, tf.float32)

                super().__init__(
                    inputs={"input_tensor": input_tensor},
                    outputs={"output_tensor": output_tensor},
                )

                self.classes = classes
                self.model_architecture = model_architecture
                self.backbone = backbone
                self.return_mask = return_mask
        else:
            raise ValueError(
                "Invalid argument for parameter `backbone`. Accepted values are a `keras_cv.models.*`, `tf.keras.models.Model` or pre-supported `keras_cv.models.VGG16`/`keras_cv.models.VGG19`"
            )

    def get_config(self):
        config = {
            "classes": self.classes,
            "backbone": self.backbone,
            "model_architecture": self.model_architecture,
            "return_mask": self.return_mask,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def FCN8S(
    classes, backbone, return_mask=False, input_shape=(None, None, 3), input_tensor=None
):
    if input_shape == (None, None, 3):
        input_shape = backbone.layers[0].input.shape
    return FullyConvolutionalNetwork(
        classes=classes,
        input_shape=input_shape,
        backbone=backbone,
        model_architecture="fcn8s",
        return_mask=return_mask,
        input_tensor=input_tensor,
    )


def FCN16S(
    classes, backbone, return_mask=False, input_shape=(None, None, 3), input_tensor=None
):
    if input_shape == (None, None, 3):
        input_shape = backbone.layers[0].input.shape
    return FullyConvolutionalNetwork(
        classes=classes,
        input_shape=input_shape,
        backbone=backbone,
        model_architecture="fcn16s",
        return_mask=return_mask,
        input_tensor=input_tensor,
    )


def FCN32S(
    classes, backbone, return_mask=False, input_shape=(None, None, 3), input_tensor=None
):
    if input_shape == (None, None, 3):
        input_shape = backbone.layers[0].input.shape

    return FullyConvolutionalNetwork(
        classes=classes,
        input_shape=input_shape,
        backbone=backbone,
        model_architecture="fcn32s",
        return_mask=return_mask,
        input_tensor=input_tensor,
    )


def FCN(
    classes, backbone, return_mask=False, input_shape=(None, None, 3), input_tensor=None
):
    if input_shape == (None, None, 3):
        input_shape = backbone.layers[0].input.shape
    return FullyConvolutionalNetwork(
        classes=classes,
        backbone=backbone,
        input_shape=input_shape,
        return_mask=return_mask,
        input_tensor=input_tensor,
    )
