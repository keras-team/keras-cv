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


from keras_cv.models.backbones.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_cv.utils.python_utils import classproperty

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


class VideoSwinTBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        include_rescaling=False,
        **kwargs,
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_tiny", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class VideoSwinSBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        include_rescaling=False,
        **kwargs,
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_small", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


class VideoSwinBBackbone(VideoSwinBackbone):
    def __new__(
        cls,
        include_rescaling=False,
        **kwargs,
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
            }
        )
        return VideoSwinBackbone.from_preset("videoswin_base", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(VideoSwinTBackbone, "__doc__", ALIAS_DOCSTRING.format(size="T"))
setattr(VideoSwinSBackbone, "__doc__", ALIAS_DOCSTRING.format(size="S"))
setattr(VideoSwinBBackbone, "__doc__", ALIAS_DOCSTRING.format(size="B"))
