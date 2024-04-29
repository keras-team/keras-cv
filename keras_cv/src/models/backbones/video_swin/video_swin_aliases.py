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

import copy

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_cv.src.models.backbones.video_swin.video_swin_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """VideoSwin{size}Backbone model.

    Reference:
        - [Video Swin Transformer](https://arxiv.org/abs/2106.13230)
        - [Video Swin Transformer GitHub](https://github.com/SwinTransformer/Video-Swin-Transformer)

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Examples:
    ```python
    input_data = np.ones(shape=(1, 32, 224, 224, 3))

    # Randomly initialized backbone
    model = VideoSwin{size}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.VideoSwinTBackbone")
class VideoSwinTBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        include_rescaling=True,
        **kwargs,
    ):
        kwargs.update(
            {
                "embed_dim": embed_dim,
                "depths": depths,
                "num_heads": num_heads,
                "window_size": window_size,
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_tiny", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "videoswin_tiny_kinetics400": copy.deepcopy(
                backbone_presets["videoswin_tiny_kinetics400"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.VideoSwinSBackbone")
class VideoSwinSBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        include_rescaling=True,
        **kwargs,
    ):
        kwargs.update(
            {
                "embed_dim": embed_dim,
                "depths": depths,
                "num_heads": num_heads,
                "window_size": window_size,
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_small", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "videoswin_small_kinetics400": copy.deepcopy(
                backbone_presets["videoswin_small_kinetics400"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.VideoSwinBBackbone")
class VideoSwinBBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 7, 7],
        include_rescaling=True,
        **kwargs,
    ):
        kwargs.update(
            {
                "embed_dim": embed_dim,
                "depths": depths,
                "num_heads": num_heads,
                "window_size": window_size,
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_base", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "videoswin_base_kinetics400": copy.deepcopy(
                backbone_presets["videoswin_base_kinetics400"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(VideoSwinTBackbone, "__doc__", ALIAS_DOCSTRING.format(size="T"))
setattr(VideoSwinSBackbone, "__doc__", ALIAS_DOCSTRING.format(size="S"))
setattr(VideoSwinBBackbone, "__doc__", ALIAS_DOCSTRING.format(size="B"))
