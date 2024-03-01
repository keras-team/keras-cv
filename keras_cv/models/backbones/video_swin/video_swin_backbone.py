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

import copy
import numpy as np
from functools import partial

from keras import layers
from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.backend import ops
from keras_cv.models import utils
from keras_cv.models.backbones.backbone import Backbone
from keras_cv.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets,
)
from keras_cv.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.utils.python_utils import classproperty

from keras_cv.layers.video_swin_layers import VideoSwinBasicLayer
from keras_cv.layers.video_swin_layers import VideoSwinPatchingAndEmbedding
from keras_cv.layers.video_swin_layers import VideoSwinPatchMerging


@keras_cv_export("keras_cv.models.VideoSwinBackbone", package="keras_cv.models")
class VideoSwinBackbone(Backbone):
    def __init__(
        self,
        *,
        include_rescaling,
        input_shape,
        input_tensor, 
        embed_dim, 
        patch_size, 
        window_size,
        mlp_ratio,
        patch_norm, 
        drop_rate,
        attn_drop_rate,
        drop_path_rate,
        depths,
        num_heads,
        qkv_bias,
        qk_scale,
        num_classes,
        **kwargs
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
            name='PatchEmbed3D'
        )(x)

        x = layers.Dropout(drop_rate, name='pos_drop')(x)
        dpr = np.linspace(0., drop_path_rate, sum(depths)).tolist()

        num_layers = len(depths)
        
        for i in range(num_layers):
            layer = VideoSwinBasicLayer(
                input_dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=VideoSwinPatchMerging if (i < num_layers - 1) else None,
                name=f'BasicLayer{i + 1}'
            )
            x = layer(x)

        x = norm_layer(axis=-1, epsilon=1e-05, name='norm')(x)
        x = layers.GlobalAveragePooling3D(name='gap3d')(x)
        output = layers.Dense(
            num_classes, use_bias=True, name='head', dtype='float32'
        )(x)
        super().__init__(inputs=input_spec, outputs=output, **kwargs)
        
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
        self.num_classes = num_classes
        self.depths = depths

    def get_config(self):
        config = super().get_config()
        config.update({
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
            "num_classes": self.num_classes,
        })
        return config
    
    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return copy.deepcopy(backbone_presets)

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return copy.deepcopy(backbone_presets_with_weights)