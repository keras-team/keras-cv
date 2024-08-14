# Copyright 2024 The KerasCV Authors
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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras


@keras_cv_export(
    "keras_cv.models.faster_rcnn.RCNNHead",
    package="keras_cv.models.faster_rcnn",
)
class RCNNHead(keras.layers.Layer):
    """A Keras layer implementing the R-CNN Head.

    Args:
        num_classes: The number of object classes to be detected.
        conv_dims: (Optional) a list of integers specifying the number of
            filters for each convolutional layer. Defaults to [].
        fc_dims: (Optional) a list of integers specifying the number of
            units for each fully-connected layer. Defaults to [1024, 1024].
    """

    def __init__(
        self,
        num_classes,
        conv_dims=[],
        fc_dims=[1024, 1024],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.convs = []
        for conv_dim in conv_dims:
            layer = keras.layers.Conv2D(
                filters=conv_dim,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="relu",
            )
            self.convs.append(layer)
        self.fcs = []
        for fc_dim in fc_dims:
            layer = keras.layers.Dense(
                units=fc_dim,
                activation="relu",
            )
            self.fcs.append(layer)
        self.box_pred = keras.layers.Dense(
            units=4,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )
        self.cls_score = keras.layers.Dense(
            units=num_classes + 1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
        )

    def call(self, feature_map, training=False):
        x = feature_map
        for conv in self.convs:
            x = conv(x, training=training)
        for fc in self.fcs:
            x = fc(x, training=training)
        rcnn_boxes = self.box_pred(x, training=training)
        rcnn_scores = self.cls_score(x, training=training)
        return rcnn_boxes, rcnn_scores

    def build(self, input_shape):
        intermediate_shape = input_shape
        if self.conv_dims:
            for idx in range(len(self.convs)):
                self.convs[idx].build(intermediate_shape)
                intermediate_shape = tuple(intermediate_shape[:-1]) + (
                    self.conv_dims[idx],
                )

        for idx in range(len(self.fc_dims)):
            self.fcs[idx].build(intermediate_shape)
            intermediate_shape = tuple(intermediate_shape[:-1]) + (
                self.fc_dims[idx],
            )

        self.box_pred.build(intermediate_shape)
        self.cls_score.build(intermediate_shape)

        self.built = True

    def get_config(self):
        config = super().get_config()
        config["num_classes"] = self.num_classes
        config["conv_dims"] = self.conv_dims
        config["fc_dims"] = self.fc_dims

        return config
