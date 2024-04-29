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

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.layers.vit_det_layers import AddPositionalEmbedding
from keras_cv.src.layers.vit_det_layers import ViTDetPatchingAndEmbedding
from keras_cv.src.layers.vit_det_layers import WindowedTransformerEncoder
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets,
)
from keras_cv.src.models.backbones.vit_det.vit_det_backbone_presets import (
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export("keras_cv.models.ViTDetBackbone", package="keras_cv.models")
class ViTDetBackbone(Backbone):
    """A ViT image encoder that uses a windowed transformer encoder and
    relative positional encodings.

    Args:
        input_shape (tuple[int], optional): The size of the input image in
            `(H, W, C)` format. Defaults to `(1024, 1024, 3)`.
        input_tensor (KerasTensor, optional): Output of
            `keras.layers.Input()`) to use as image input for the model.
            Defaults to `None`.
        include_rescaling (bool, optional): Whether to rescale the inputs. If
            set to `True`, inputs will be passed through a
            `Rescaling(1/255.0)` layer. Defaults to `False`.
        patch_size (int, optional): the patch size to be supplied to the
            Patching layer to turn input images into a flattened sequence of
            patches. Defaults to `16`.
        embed_dim (int, optional): The latent dimensionality to be projected
            into in the output of each stacked windowed transformer encoder.
            Defaults to `768`.
        depth (int, optional): The number of transformer encoder layers to
            stack in the Vision Transformer. Defaults to `12`.
        mlp_dim (int, optional): The dimensionality of the hidden Dense
            layer in the transformer MLP head. Defaults to `768*4`.
        num_heads (int, optional): the number of heads to use in the
            `MultiHeadAttentionWithRelativePE` layer of each transformer
            encoder. Defaults to `12`.
        out_chans (int, optional): The number of channels (features) in the
            output (image encodings). Defaults to `256`.
        use_bias (bool, optional): Whether to use bias to project the keys,
            queries, and values in the attention layer. Defaults to `True`.
        use_abs_pos (bool, optional): Whether to add absolute positional
            embeddings to the output patches. Defaults to `True`.
        use_rel_pos (bool, optional): Whether to use relative positional
            emcodings in the attention layer. Defaults to `True`.
        window_size (int, optional): The size of the window for windowed
            attention in the transformer encoder blocks. Defaults to `14`.
        global_attention_indices (list, optional): Indexes for blocks using
            global attention. Defaults to `[2, 5, 8, 11]`.
        layer_norm_epsilon (int, optional): The epsilon to use in the layer
            normalization blocks in transformer encoder. Defaults to `1e-6`.

    References:
        - [Segment Anything paper](https://arxiv.org/abs/2304.02643)
        - [Segment Anything GitHub](https://github.com/facebookresearch/segment-anything)
        - [Detectron2](https://github.com/facebookresearch/detectron2)
    """  # noqa: E501

    def __init__(
        self,
        *,
        include_rescaling,
        input_shape=(1024, 1024, 3),
        input_tensor=None,
        patch_size=16,
        embed_dim=768,
        depth=12,
        mlp_dim=768 * 4,
        num_heads=12,
        out_chans=256,
        use_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        global_attention_indices=[2, 5, 8, 11],
        layer_norm_epsilon=1e-6,
        **kwargs
    ):
        img_input = utils.parse_model_inputs(
            input_shape, input_tensor, name="images"
        )

        # Check that the input image is well specified.
        if img_input.shape[-3] is None or img_input.shape[-2] is None:
            raise ValueError(
                "Height and width of the image must be specified"
                " in `input_shape`."
            )
        if img_input.shape[-3] != img_input.shape[-2]:
            raise ValueError(
                "Input image must be square i.e. the height must"
                " be equal to the width in the `input_shape`"
                " tuple/tensor."
            )

        img_size = img_input.shape[-3]

        x = img_input

        if include_rescaling:
            # Use common rescaling strategy across keras_cv
            x = keras.layers.Rescaling(1.0 / 255.0)(x)

        # VITDet scales inputs based on the standard ImageNet mean/stddev.
        x = (x - ops.array([0.485, 0.456, 0.406], dtype=x.dtype)) / (
            ops.array([0.229, 0.224, 0.225], dtype=x.dtype)
        )

        x = ViTDetPatchingAndEmbedding(
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            embed_dim=embed_dim,
        )(x)
        if use_abs_pos:
            x = AddPositionalEmbedding(img_size, patch_size, embed_dim)(x)

        for i in range(depth):
            x = WindowedTransformerEncoder(
                project_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                use_bias=use_bias,
                use_rel_pos=use_rel_pos,
                window_size=(
                    window_size if i not in global_attention_indices else 0
                ),
                input_size=(img_size // patch_size, img_size // patch_size),
            )(x)
        x = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    filters=out_chans, kernel_size=1, use_bias=False
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Conv2D(
                    filters=out_chans,
                    kernel_size=3,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.LayerNormalization(epsilon=1e-6),
            ]
        )(x)

        super().__init__(inputs=img_input, outputs=x, **kwargs)

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.out_chans = out_chans
        self.use_bias = use_bias
        self.use_rel_pos = use_rel_pos
        self.use_abs_pos = use_abs_pos
        self.window_size = window_size
        self.global_attention_indices = global_attention_indices
        self.layer_norm_epsilon = layer_norm_epsilon
        self.input_tensor = input_tensor
        self.include_rescaling = include_rescaling

    @property
    def pyramid_level_inputs(self):
        raise NotImplementedError(
            "The `ViTDetBackbone` model doesn't compute"
            " pyramid level features."
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
                "include_rescaling": self.include_rescaling,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
                "depth": self.depth,
                "mlp_dim": self.mlp_dim,
                "num_heads": self.num_heads,
                "out_chans": self.out_chans,
                "use_bias": self.use_bias,
                "use_abs_pos": self.use_abs_pos,
                "use_rel_pos": self.use_rel_pos,
                "window_size": self.window_size,
                "global_attention_indices": self.global_attention_indices,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
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
