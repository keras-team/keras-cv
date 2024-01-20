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
from keras_cv.backend import ops

from keras import layers

class MLP(layers.Layer):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop_rate=0.0,
        act_layer=layers.Activation("gelu"),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.drop_rate = drop_rate
        self.act = act_layer
        self.fc1 = layers.Dense(self.hidden_features)
        self.fc2 = layers.Dense(self.out_features)
        self.dropout = layers.Dropout(self.drop_rate)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_features": self.out_features, 
                "hidden_features": self.hidden_features,
                "drop_rate": self.drop_rate,
            }
        )
        return config