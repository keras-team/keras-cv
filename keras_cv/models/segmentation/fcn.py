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

from keras_cv.layers.preprocessing import Resizing
from keras_cv.models import VGG16
from keras_cv.models import VGG19
import numpy as np

BACKBONE_CONFIG = {
    "vgg16": {"BLOCK3": 11, "BLOCK4": 15, "BLOCK5": 19, "DENSE_START": -3},
    "vgg19": {"BLOCK3": 12, "BLOCK4": 17, "BLOCK5": 22, "DENSE_START": -3},
}

class BilinearInitializer(tf.keras.initializers.Initializer):
    '''Initializer for Conv2DTranspose to perform bilinear interpolation on each channel.'''
    def __call__(self, shape, dtype=None, **kwargs):
        kernel_size, _, filters, _ = shape
        arr = np.zeros((kernel_size, kernel_size, filters, filters))
        ## make filter that performs bilinear interpolation through Conv2DTranspose
        upscale_factor = (kernel_size+1)//2
        if kernel_size % 2 == 1:
            center = upscale_factor - 1
        else:
            center = upscale_factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        kernel = (1-np.abs(og[0]-center)/upscale_factor) * \
                 (1-np.abs(og[1]-center)/upscale_factor) # kernel shape is (kernel_size, kernel_size)
        for i in range(filters):
            arr[..., i, i] = kernel
        return tf.convert_to_tensor(arr, dtype=dtype)
    
class FCN(tf.keras.models.Model):
    """A segmentation model based on the Fully Convolutional Network introduced by Long et. al.

    Args:
        classes: int, the number of classes for the segmentation model. Note that
            the classes doesn't contain the background class, and the classes
            from the data should be represented by integers with range
            [0, classes).
        backbone: a backbone network for the model. Can be a `tf.keras.models.Model`
            instance. The supported pre-defined backbone models are:
            1. "vgg16", a VGG16 model
            2. "vgg19", a VGG19 model
            Defaults to 'vgg16'.
            Note: If a custom `tf.keras.models.Model` is passed, then only the
            `Conv2D`, `MaxPooling2D` and `Dense` layers are extracted from it
            to make the custom backbone.
        model_architecture: str, defines the model architecture based on the
            implementation details present in the paper. The supported
            architectures are:
            1. 'fcn8s', a FCN-8S definition
            2. 'fcn16s', a FCN-16S definition
            3. 'fcn32s', a FCN-32S definition
            Defaults to 'fcn8s'.
        include_resizing: bool, defines whether to include a `tf.keras.layers.Resizing` layer which defaults to a height and width of value 224. Defaults to `False`

    """

    def __init__(
        self,
        classes,
        backbone="vgg16",
        model_architecture=None,
        include_resizing=False,
        **kwargs
    ):
        super(FCN, self).__init__(**kwargs)

        self.resize = include_resizing
        self.num_classes = classes
        self.backbone = backbone
        self.model_architecture = model_architecture

        if self.resize:
            self.resizing_layer = Resizing(
                height=224, width=224, interpolation="bilinear"
            )

    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]

        if isinstance(self.backbone, str):

            if self.backbone == "vgg16":
                # Assumes `channels_last` format
                if len(input_shape) == 3:
                    self.vgg_input_shape = (
                        input_shape[0],
                        input_shape[1],
                        input_shape[2],
                    )
                elif len(input_shape) == 4:
                    self.vgg_input_shape = (
                        input_shape[1],
                        input_shape[2],
                        input_shape[3],
                    )
                self.backbone_base_model = VGG16(
                    include_rescaling=False,
                    include_top=True,
                    classes=self.num_classes,
                    input_shape=self.vgg_input_shape,
                )
                self.backbone_name = self.backbone
            elif self.backbone == "vgg19":
                self.vgg_input_shape = (input_shape[1], input_shape[2], input_shape[3])
                self.backbone_base_model = VGG19(
                    include_rescaling=False,
                    include_top=True,
                    classes=self.num_classes,
                    input_shape=self.vgg_input_shape,
                )
                self.backbone_name = self.backbone
            else:
                raise ValueError(
                    "Entered `backbone` argument is not a standard available backbone. Valid options are ['vgg16', 'vgg19']"
                )

        else:
            if isinstance(self.backbone, tf.keras.models.Model):
                layer_list = []
                # Design choice : Either fully reject, or pick up only Conv2D and MaxPooling2D layers and convert Dense to Conv in `build()`
                # Possible edge case : ResNets
                # Current design allows a simple CNN with Conv2D, MaxPooling2D and Dense layers only, to be parsed and made into a FCN directly.
                for i in self.backbone.layers:
                    if isinstance(i, tf.keras.layers.Conv2D) or isinstance(
                        tf.keras.layers.MaxPooling2D
                    ):
                        layer_list.append(i)
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
                        "Entered `backbone` argument has custom layers. Include a `tf.keras.models.Model` with `tf.keras.layers.Conv2D` or `tf.keras.layers.MaxPooling2D` layers only."
                    )
                self.backbone = tf.keras.Sequential(layer_list)
                self.backbone_base_model = self.backbone
            else:
                raise ValueError(
                    "Unsupported type. Valid types are `tf.keras.models.Model`"
                )

        if self.model_architecture not in ["fcn8s", "fcn16s", "fcn32s"]:
            raise ValueError(
                "Entered `model_architecture` argument is not a standard available model_architecture. Valid options are ['fcn8s', 'fcn16s', 'fcn32s']"
            )
        elif not isinstance(self.backbone, str) and isinstance(
            self.model_architecture, None
        ):
            raise ValueError(
                "Using `model_architecture` is not allowed when supplying a custom backbone. Valid options are ['fcn8s', 'fcn16s', 'fcn32s']"
            )
        else:
            if self.model_architecture is None:
                self.model_architecture = "fcn8s"
            if self.model_architecture == "fcn8s":
                self.pool3 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK3"]
                    ]
                )
                self.pool4 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK4"]
                    ]
                )
                self.pool5 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                    ]
                )
            elif self.model_architecture == "fcn16s":
                self.pool4 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK4"]
                    ]
                )
                self.pool5 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                    ]
                )
            elif self.model_architecture == "fcn32s":
                self.pool5 = tf.keras.Sequential(
                    self.backbone_base_model.layers[
                        : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                    ]
                )

        if self.model_architecture is not None:
            units = [
                i.units
                for i in [
                    self.backbone_base_model.layers[
                        BACKBONE_CONFIG[self.backbone_name]["DENSE_START"]
                    ]
                ]
            ]
            self.dense_convs = [
                tf.keras.layers.Conv2D(
                    filters=i,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    activation="relu",
                    padding="same",
                )
                for i in units
            ]
        else:
            units = [
                i.units
                for i in self.backbone_base_model.layers
                if isinstance(i, tf.keras.layers.Dense)
            ]
            if len(units) != 0:
                self.dense_convs = []
                for idx in len(units):
                    curr_unit = units[idx]
                    self.dense_convs.append(
                        tf.keras.layers.Conv2D(
                            filters=curr_unit,
                            strides=(1, 1),
                            activation="relu",
                            padding="same",
                        )
                    )
                    if idx != len(units) - 1:
                        self.dense_convs.append(tf.keras.layers.Dropout(0.5))
            output_shape = self.dense_convs[-1].compute_output_shape()
            target_height_factor = self.height // output_shape[1]
            target_width_factor = self.width // output_shape[2]
            if output_shape[3] != self.num_classes:
                self.dense_convs.append(
                    tf.keras.layers.Conv2D(
                        filters=self.num_classes,
                        strides=(1, 1),
                        activation="relu",
                        padding="same",
                    )
                )
            self.output_layer = tf.keras.layers.UpSampling2D(
                size=(target_height_factor, target_width_factor),
                data_format="channels_last",
                interpolation="bilinear",
            )

    def call(self, x):
        if self.resize:
            x = self.resizing_layer(x)

        if self.model_architecture == "fcn8s":
            pool3_output = self.pool3(x)
            pool4_output = self.pool4(x)
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            pool5_output = tf.keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(pool5_output)

            pool3_output = tf.keras.layers.Conv2D(
                filters=pool5_output.shape[-1],
                kernel_size=(1, 1),
                padding="same",
                strides=(1, 1),
            )(pool3_output)
            pool4_output = tf.keras.layers.Conv2D(
                filters=pool5_output.shape[-1],
                kernel_size=(1, 1),
                padding="same",
                strides=(1, 1),
            )(pool4_output)

            intermediate_pool_output = tf.keras.layers.Add()(
                [pool4_output, pool5_output]
            )
            intermediate_pool_output = tf.keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(intermediate_pool_output)

            final_pool_output = tf.keras.layers.Add()(
                [pool3_output, intermediate_pool_output]
            )

            output_layer = tf.keras.layers.UpSampling2D(
                size=(8, 8), data_format="channels_last", interpolation="bilinear"
            )
            return output_layer(final_pool_output)

        elif self.model_architecture == "fcn16s":
            pool4_output = self.pool4(x)
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            pool5_output = tf.keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(pool5_output)

            pool4_output = tf.keras.layers.Conv2D(
                filters=pool5_output.shape[-1],
                kernel_size=(1, 1),
                padding="same",
                strides=(1, 1),
            )(pool4_output)

            final_pool_output = tf.keras.layers.Add()(
                [pool4_output, intermediate_pool_output]
            )

            output_layer = tf.keras.layers.UpSampling2D(
                size=(16, 16), data_format="channels_last", interpolation="bilinear"
            )
            return output_layer(final_pool_output)

        elif self.model_architecture == "fcn32s":
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            output_layer = tf.keras.layers.UpSampling2D(
                size=(32, 32), data_format="channels_last", interpolation="bilinear"
            )
            return output_layer(pool5_output)

        else:
            x = self.backbone(x)
            for layer in self.dense_convs:
                x = layer(x)
            return self.output_layer(x)

    def train_step(self, data):
        images, y_true, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self.call(images)
            total_loss = self.compute_loss(
                x=images, y=y_true, y_pred=y_pred, sample_weight=sample_weight
            )
        self.optimizer.minimize(total_loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(
            x=images, y=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def get_config(self):
        config = {
            "classes": self.num_classes,
            "backbone": self.backbone,
            "model_architecture": self.model_architecture,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
