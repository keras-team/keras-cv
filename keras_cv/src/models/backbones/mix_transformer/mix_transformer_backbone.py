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

"""MiT backbone model.

References:
  - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) (CVPR 2021)
  - [Based on the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/blob/main/deepvision/models/classification/mix_transformer/mit_tf.py)
  - [Based on the NVlabs' official PyTorch implementation](https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/backbones/mix_transformer.py)
  - [Inspired by @sithu31296's reimplementation](https://github.com/sithu31296/semantic-segmentation/blob/main/semseg/models/backbones/mit.py)
"""  # noqa: E501

import copy

import numpy as np

from keras_cv.src import layers as cv_layers
from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.backend import ops
from keras_cv.src.models import utils
from keras_cv.src.models.backbones.backbone import Backbone
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets_with_weights,
)
from keras_cv.src.utils.python_utils import classproperty


@keras_cv_export("keras_cv.models.MiTBackbone")
class MiTBackbone(Backbone):
    def __init__(
        self,
        include_rescaling,
        depths,
        input_shape=(224, 224, 3),
        input_tensor=None,
        embedding_dims=None,
        **kwargs,
    ):
        """A Keras model implementing the MixTransformer architecture to be
        used as a backbone for the SegFormer architecture.

        References:
            - [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) # noqa: E501
            - [Based on the TensorFlow implementation from DeepVision](https://github.com/DavidLandup0/deepvision/tree/main/deepvision/models/classification/mix_transformer) # noqa: E501

        Args:
            include_rescaling: bool, whether to rescale the inputs. If set
                to `True`, inputs will be passed through a `Rescaling(1/255.0)`
                layer.
            depths: the number of transformer encoders to be used per stage in the
                network
            embedding_dims: the embedding dims per hierarchical stage, used as
                the levels of the feature pyramid
            input_shape: optional shape tuple, defaults to (None, None, 3).
            input_tensor: optional Keras tensor (i.e. output of `keras.layers.Input()`)
                to use as image input for the model.

        Example:

        Using the class with a `backbone`:

        ```python
        import tensorflow as tf
        import keras_cv

        images = np.ones(shape=(1, 96, 96, 3))
        labels = np.zeros(shape=(1, 96, 96, 1))
        backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")

        # Evaluate model
        model(images)

        # Train model
        model.compile(
            optimizer="adam",
            loss=keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        model.fit(images, labels, epochs=3)
        ```
        """
        drop_path_rate = 0.1
        dpr = [x for x in np.linspace(0.0, drop_path_rate, sum(depths))]
        blockwise_num_heads = [1, 2, 5, 8]
        blockwise_sr_ratios = [8, 4, 2, 1]
        num_stages = 4

        cur = 0
        patch_embedding_layers = []
        transformer_blocks = []
        layer_norms = []

        for i in range(num_stages):
            patch_embed_layer = cv_layers.OverlappingPatchingAndEmbedding(
                project_dim=embedding_dims[0] if i == 0 else embedding_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                name=f"patch_and_embed_{i}",
            )
            patch_embedding_layers.append(patch_embed_layer)

            transformer_block = [
                cv_layers.HierarchicalTransformerEncoder(
                    project_dim=embedding_dims[i],
                    num_heads=blockwise_num_heads[i],
                    sr_ratio=blockwise_sr_ratios[i],
                    drop_prob=dpr[cur + k],
                    name=f"hierarchical_encoder_{i}_{k}",
                )
                for k in range(depths[i])
            ]
            transformer_blocks.append(transformer_block)
            cur += depths[i]
            layer_norms.append(keras.layers.LayerNormalization())

        inputs = utils.parse_model_inputs(input_shape, input_tensor)
        x = inputs

        if include_rescaling:
            x = keras.layers.Rescaling(scale=1 / 255)(x)

        pyramid_level_inputs = []
        for i in range(num_stages):
            # Compute new height/width after the `proj`
            # call in `OverlappingPatchingAndEmbedding`
            stride = 4 if i == 0 else 2
            new_height, new_width = (
                int(ops.shape(x)[1] / stride),
                int(ops.shape(x)[2] / stride),
            )

            x = patch_embedding_layers[i](x)
            for blk in transformer_blocks[i]:
                x = blk(x)
            x = layer_norms[i](x)
            x = keras.layers.Reshape(
                (new_height, new_width, -1), name=f"output_level_{i}"
            )(x)
            pyramid_level_inputs.append(utils.get_tensor_input_name(x))

        super().__init__(inputs=inputs, outputs=x, **kwargs)

        self.depths = depths
        self.embedding_dims = embedding_dims
        self.include_rescaling = include_rescaling
        self.input_tensor = input_tensor
        self.pyramid_level_inputs = {
            f"P{i + 1}": name for i, name in enumerate(pyramid_level_inputs)
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embedding_dims": self.embedding_dims,
                "include_rescaling": self.include_rescaling,
                "input_shape": self.input_shape[1:],
                "input_tensor": self.input_tensor,
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
