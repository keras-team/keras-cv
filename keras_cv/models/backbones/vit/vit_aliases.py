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

from keras_cv.models.backbones.vit.vit_backbone import ViTBackbone
from keras_cv.models.backbones.vit.vit_backbone_presets import backbone_presets
from keras_cv.utils.python_utils import classproperty


class ViTTiny16Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vittiny16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vittiny16_imagenet": copy.deepcopy(
                backbone_presets["vittiny16_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTS16Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vits16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vits16_imagenet": copy.deepcopy(
                backbone_presets["vits16_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTB16Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vitb16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitb16_imagenet": copy.deepcopy(
                backbone_presets["vitb16_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTL16Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vitl16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitl16_imagenet": copy.deepcopy(
                backbone_presets["vitl16_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTH16Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vith16", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return {}


class ViTTiny32Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vittiny32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return {}


class ViTS32Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vits32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vits32_imagenet": copy.deepcopy(
                backbone_presets["vits32_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTB32Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vitb32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "vitb32_imagenet": copy.deepcopy(
                backbone_presets["vitb32_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return cls.presets


class ViTL32Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vitl32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return {}


class ViTH32Backbone(ViTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs
    ):
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return ViTBackbone.from_preset("vith32", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """
        Dictionary of preset names and configurations that include weights.
        """
        return {}


ALIAS_DOCSTRING = """ViTBackbone Model with the {name} configuration for {patch_size} patch size.
    Reference:
        - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929v2)
        (ICLR 2021)
    This function returns a Keras {name} model.

    The naming convention of ViT models follows: ViTSize_Patch-size
        (i.e. ViTS16).
    The following sizes were released in the original paper:
        - S (Small)
        - B (Base)
        - L (Large)
    But subsequent work from the same authors introduced:
        - Ti (Tiny)
        - H (Huge)

    The parameter configurations for all of these sizes, at patch sizes 16 and
    32 are made available, following the naming convention laid out above.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).
    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(scale=1./255.0)`
            layer. Note that ViTs expect an input range of `[0..1]` if rescaling
            isn't used. Regardless of whether you supply `[0..1]` or the input
            is rescaled to `[0..1]`, the inputs will further be rescaled to
            `[-1..1]`.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
"""  # noqa: E501

setattr(
    ViTTiny16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTTiny16", patch_size=16),
)
setattr(
    ViTS16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTS16", patch_size=16),
)
setattr(
    ViTB16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTB16", patch_size=16),
)
setattr(
    ViTL16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTL16", patch_size=16),
)
setattr(
    ViTH16Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTH16", patch_size=16),
)
setattr(
    ViTTiny32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTTiny32", patch_size=32),
)
setattr(
    ViTS32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTS32", patch_size=32),
)
setattr(
    ViTB32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTB32", patch_size=32),
)
setattr(
    ViTL32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTL32", patch_size=32),
)
setattr(
    ViTH32Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="ViTH32", patch_size=32),
)
