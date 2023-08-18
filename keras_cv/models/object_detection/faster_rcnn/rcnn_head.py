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

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras


@keras_cv_export(
    "keras_cv.models.faster_rcnn.RCNNHead",
    package="keras_cv.models.faster_rcnn",
)
class RCNNHead(keras.layers.Layer):
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
            layer = keras.layers.Dense(units=fc_dim, activation="relu")
            self.fcs.append(layer)
        self.box_pred = keras.layers.Dense(units=4)
        self.cls_score = keras.layers.Dense(
            units=num_classes + 1, activation="softmax"
        )

    def call(self, feature_map, training=None):
        x = feature_map
        for conv in self.convs:
            x = conv(x)
        for fc in self.fcs:
            x = fc(x)
        rcnn_boxes = self.box_pred(x)
        rcnn_scores = self.cls_score(x)
        return rcnn_boxes, rcnn_scores

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "conv_dims": self.conv_dims,
            "fc_dims": self.fc_dims,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
