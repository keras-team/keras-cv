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
    

class PatchEmbed3D(keras.Model):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (keras.layers, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), embed_dim=96, norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.norm_layer = norm_layer

    def build(self, input_shape):
        self.pads = [
            [0, 0],
            self._compute_padding(input_shape[1], self.patch_size[0]),
            self._compute_padding(input_shape[2], self.patch_size[1]),
            self._compute_padding(input_shape[3], self.patch_size[2]),
            [0, 0],
        ]

        # layers
        self.proj = layers.Conv3D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            name="embed_proj",
        )
        if self.norm_layer is not None:
            self.norm = self.norm_layer(axis=-1, epsilon=1e-5, name="embed_norm")
        else:
            self.norm = None

    def _compute_padding(self, dim, patch_size):
        pad_amount = patch_size - (dim % patch_size)
        return [0, pad_amount if pad_amount != patch_size else 0]

    def compute_output_shape(self, input_shape):
        spatial_dims = [
            (dim - self.patch_size[i]) // self.patch_size[i] + 1
            for i, dim in enumerate(input_shape[1:-1])
        ]
        output_shape = (input_shape[0],) + tuple(spatial_dims) + (self.embed_dim,)
        return output_shape

    def call(self, x):
        x = ops.pad(x, self.pads)
        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)

        return x