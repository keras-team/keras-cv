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

BACKBONE_CONFIG = {
    "vgg16": {"BLOCK3": 11, "BLOCK4": 15, "BLOCK5": 19, "DENSE_START": -3},
    "vgg19": {"BLOCK3": 12, "BLOCK4": 17, "BLOCK5": 22, "DENSE_START": -3},
}


class FCN(tf.keras.models.Model):
    def _init_(self, classes, backbone="vgg16", model_architecture=None, **kwargs):
        super(FCN, self)._init_(**kwargs)
        # TODO: Perform error handling for all params
        # backbone done
        # if backbone is custom, no model arch allowed

        if isinstance(backbone, str):

            if backbone == "vgg16":
                self.backbone_base_model = keras.applications.VGG16()
                self.backbone_name = backbone
            elif backbone == "vgg19":
                self.backbone_base_model = keras.applications.VGG19()
                self.backbone_name = backbone
            else:
                raise ValueError(
                    "Entered `backbone` argument is not a standard available backbone. Valid options are ['vgg16', 'vgg19']"
                )

        else:
            if isinstance(backbone, tf.keras.models.Model):
                layer_list = []
                # Design choice : Either fully reject, or pick up only Conv2D and MaxPooling2D layers and convert Dense to Conv in `build()`
                # Possible edge case : ResNets
                # Current design allows a simple CNN with Conv2D, MaxPooling2D and Dense layers only to be parsed and made into a FCN directly.
                for i in backbone.layers:
                    if isinstance(i, keras.layers.Conv2D) or isinstance(
                        keras.layers.MaxPooling2D
                    ):
                        layer_list.append(i)
                if len(layer_list) == 0:
                    raise ValueError(
                        "Entered `backbone` argument does not have any Conv2D or MaxPooling layers. Include a `tf.keras.models.Model` with `keras.layers.Conv2D` or `keras.layers.MaxPooling2D` layers"
                    )
                if not all(
                    isinstance(layer, keras.layers.Conv2D)
                    or isinstance(layer, keras.layers.MaxPooling2D)
                    or isinstance(layer, keras.layers.Dense)
                    for layer in layer_list
                ):
                    raise ValueError(
                        "Entered `backbone` argument has custom layers. Include a `tf.keras.models.Model` with `keras.layers.Conv2D` or `keras.layers.MaxPooling2D` layers only."
                    )
                self.backbone = keras.Sequential(layer_list)
                self.backbone_base_model = backbone
            else:
                raise ValueError(
                    "Unsupported type. Valid types are `tf.keras.model.Model`"
                )

        if model_architecture not in ["fcn8s", "fcn16s", "fcn32s"]:
            raise ValueError(
                "Entered `model_architecture` argument is not a standard available model_architecture. Valid options are ['fcn8s', 'fcn16s', 'fcn32s']"
            )
        elif not isinstance(backbone, str) and isinstance(model_architecture, None):
            raise ValueError(
                "Using `model_architecture` is not allowed when supplying a custom backbone. Valid options are ['fcn8s', 'fcn16s', 'fcn32s']"
            )
        else:
            self.model_architecture = model_architecture
            if model_architecture == "fcn8s":
                self.pool3 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK3"]
                ]
                self.pool4 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK4"]
                ]
                self.pool5 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                ]
            elif model_architecture == "fcn16s":
                self.pool4 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK4"]
                ]
                self.pool5 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                ]
            elif model_architecture == "fcn32s":
                self.pool5 = self.backbone_base_model.layers[
                    : BACKBONE_CONFIG[self.backbone_name]["BLOCK5"]
                ]

        self.num_classes = classes

    def build(self, input_shape):
        self.height = input_shape[1]
        self.width = input_shape[2]

        if self.model_architecture is not None:
            units = [
                i.units
                for i in self.backbone_base_model.layers[
                    BACKBONE_CONFIG[self.backbone_name]["DENSE_START"]
                ]
            ]
            self.dense_convs = [
                keras.layers.Conv2D(
                    filters=i, strides=(1, 1), activation="relu", padding="same"
                )
                for i in units
            ]
        else:
            units = [
                i.units
                for i in self.backbone_base_model.layers
                if isinstance(i, keras.layers.Dense)
            ]
            if len(units) != 0:
                self.dense_convs = [
                    keras.layers.Conv2D(
                        filters=i, strides=(1, 1), activation="relu", padding="same"
                    )
                    for i in units
                ]
            output_shape = self.dense_convs[-1].compute_output_shape()
            target_height_factor = self.height // output_shape[1]
            target_width_factor = self.width // output_shape[2]
            if output_shape[3] != self.num_classes:
                self.dense_convs.append(
                    keras.layers.Conv2D(
                        filters=self.num_classes,
                        strides=(1, 1),
                        activation="relu",
                        padding="same",
                    )
                )
            self.output_layer = keras.layers.UpSampling2D(
                size=(target_height_factor, target_width_factor),
                data_format="channels_last",
                interpolation="bilinear",
            )

    def call(self, x):
        if self.model_architecture == "fcn8s":
            pool3_output = self.pool3(x)
            pool4_output = self.pool4(x)
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            pool5_output = keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(pool5_output)

            intermediate_pool_output = keras.layers.Add()[pool4_output, pool5_output]
            intermediate_pool_output = keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(intermediate_pool_output)

            final_pool_output = keras.layers.Add()[
                pool3_output, intermediate_pool_output
            ]

            output_layer = keras.layers.UpSampling2D(
                size=(8, 8), data_format="channels_last", interpolation="bilinear"
            )
            return output_layer(final_pool_output)

        elif self.model_architecture == "fcn16s":
            pool4_output = self.pool4(x)
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            pool5_output = keras.layers.UpSampling2D(
                size=(2, 2), data_format="channels_last", interpolation="bilinear"
            )(pool5_output)

            final_pool_output = keras.layers.Add()[
                pool4_output, intermediate_pool_output
            ]

            output_layer = keras.layers.UpSampling2D(
                size=(16, 16), data_format="channels_last", interpolation="bilinear"
            )
            return output_layer(final_pool_output)

        elif self.model_architecture == "fcn32s":
            pool5_output = self.pool5(x)

            for layer in self.dense_convs:
                pool5_output = layer(pool5_output)
            output_layer = keras.layers.UpSampling2D(
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
