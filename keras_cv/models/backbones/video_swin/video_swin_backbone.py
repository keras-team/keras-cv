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

from functools import partial

import numpy as np
from keras import layers

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.layers.video_swin_layers import VideoSwinBasicLayer
from keras_cv.layers.video_swin_layers import VideoSwinPatchingAndEmbedding
from keras_cv.layers.video_swin_layers import VideoSwinPatchMerging
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone


@keras_cv_export("keras_cv.models.VideoSwinBackbone", package="keras_cv.models")
class VideoSwinBackbone(Backbone):
    def __init__(
        self,
        *,
        include_rescaling,
        input_shape=(32, 224, 224, 3),
        input_tensor=None,
        embed_dim=96,
        patch_size=[2, 4, 4],
        window_size=[8, 7, 7],
        mlp_ratio=4.0,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        qkv_bias=True,
        qk_scale=None,
        **kwargs,
    ):
        input_spec = utils.parse_model_inputs(
            input_shape, input_tensor, name="videos"
        )

        # Check that the input video is well specified.
        if input_spec.shape[-3] is None or input_spec.shape[-2] is None:
            raise ValueError(
                "Height and width of the video must be specified"
                " in `input_shape`."
            )
        if input_spec.shape[-3] != input_spec.shape[-2]:
            raise ValueError(
                "Input video must be square i.e. the height must"
                " be equal to the width in the `input_shape`"
                " tuple/tensor."
            )

        x = input_spec

        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = keras.layers.Rescaling(1.0 / 255.0)(x)

        norm_layer = partial(layers.LayerNormalization, epsilon=1e-05)

        x = VideoSwinPatchingAndEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
            name="videoswin_patching_and_embedding",
        )(x)
        x = layers.Dropout(drop_rate, name="pos_drop")(x)

        dpr = np.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        num_layers = len(depths)
        for i in range(num_layers):
            layer = VideoSwinBasicLayer(
                input_dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=VideoSwinPatchMerging
                if (i < num_layers - 1)
                else None,
                name=f"videoswin_basic_layer_{i + 1}",
            )
            x = layer(x)

        x = norm_layer(axis=-1, epsilon=1e-05, name="norm")(x)
        super().__init__(inputs=input_spec, outputs=x, **kwargs)

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.depths = depths

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "patch_norm": self.patch_norm,
                "window_size": self.window_size,
                "patch_size": self.patch_size,
                "mlp_ratio": self.mlp_ratio,
                "drop_rate": self.drop_rate,
                "drop_path_rate": self.drop_path_rate,
                "attn_drop_rate": self.attn_drop_rate,
                "depths": self.depths,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "qk_scale": self.qk_scale,
            }
        )
        return config
