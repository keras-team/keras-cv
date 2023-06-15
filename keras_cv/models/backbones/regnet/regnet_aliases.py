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

from keras_cv.models.backbones.regnet.regnet_backbone import (  # noqa: E501
    RegNetBackbone,
)
from keras_cv.models.backbones.regnet.regnetx_backbone_presets import (  # noqa: E501
    backbone_presets_x,
)
from keras_cv.models.backbones.regnet.regnety_backbone_presets import (  # noqa: E501
    backbone_presets_y,
)
from keras_cv.utils.python_utils import classproperty

ALIAS_DOCSTRING = """This class represents the {name} architecture.

  Reference:
    - [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)
    (CVPR 2020)

  For image classification use cases, see [this page for detailed examples](https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).


  The naming of models is as follows: `RegNet<block_type><flops>` where
  `block_type` is one of `(X, Y)` and `flops` signifies hundred million
  floating point operations. For example RegNetY064 corresponds to RegNet with
  Y block and 6.4 giga flops (64 hundred million flops).

  Args:
    include_rescaling: whether or not to Rescale the inputs.If set to True,
        inputs will be passed through a `Rescaling(1/255.0)` layer.
    input_tensor: Optional Keras tensor (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, defaults to (None, None, 3).
        It should have exactly 3 inputs channels.
"""  # noqa: E501


# Instantiating variants
class RegNetX002Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx002", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx002": copy.deepcopy(backbone_presets_x["regnetx002"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX004Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx004", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx004": copy.deepcopy(backbone_presets_x["regnetx004"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX006Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx006", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx006": copy.deepcopy(backbone_presets_x["regnetx006"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX008Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx008", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx008": copy.deepcopy(backbone_presets_x["regnetx008"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX016Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx016", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx016": copy.deepcopy(backbone_presets_x["regnetx016"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX032Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx032", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx032": copy.deepcopy(backbone_presets_x["regnetx032"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX040Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx040", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx040": copy.deepcopy(backbone_presets_x["regnetx040"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX064Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx064", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx064": copy.deepcopy(backbone_presets_x["regnetx064"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX080Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx080", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx080": copy.deepcopy(backbone_presets_x["regnetx080"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX120Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx120", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx120": copy.deepcopy(backbone_presets_x["regnetx120"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX160Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx160", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx160": copy.deepcopy(backbone_presets_x["regnetx160"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetX320Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnetx320", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnetx320": copy.deepcopy(backbone_presets_x["regnetx320"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY002Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety002", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety002": copy.deepcopy(backbone_presets_y["regnety002"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY004Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety004", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety004": copy.deepcopy(backbone_presets_y["regnety004"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY006Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety006", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety006": copy.deepcopy(backbone_presets_y["regnety006"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY008Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety008", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety008": copy.deepcopy(backbone_presets_y["regnety008"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY016Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety016", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety016": copy.deepcopy(backbone_presets_y["regnety016"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY032Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety032", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety032": copy.deepcopy(backbone_presets_y["regnety032"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY040Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety040", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety040": copy.deepcopy(backbone_presets_y["regnety040"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY064Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety064", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety064": copy.deepcopy(backbone_presets_y["regnety064"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY080Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety080", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety080": copy.deepcopy(backbone_presets_y["regnety080"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY120Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety120", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety120": copy.deepcopy(backbone_presets_y["regnety120"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY160Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety160", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety160": copy.deepcopy(backbone_presets_y["regnety160"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


class RegNetY320Backbone(RegNetBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_tensor=None,
        input_shape=(None, None, 3),
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_tensor": input_tensor,
                "input_shape": input_shape,
            }
        )
        return RegNetBackbone.from_preset("regnety320", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "regnety320": copy.deepcopy(backbone_presets_y["regnety320"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


RegNetX002Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX002Backbone")
RegNetX004Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX004Backbone")
RegNetX006Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX006Backbone")
RegNetX008Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX008Backbone")
RegNetX016Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX016Backbone")
RegNetX032Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX032Backbone")
RegNetX040Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX040Backbone")
RegNetX064Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX064Backbone")
RegNetX080Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX080Backbone")
RegNetX120Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX120Backbone")
RegNetX160Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX160Backbone")
RegNetX320Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetX320Backbone")

RegNetY002Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY002Backbone")
RegNetY004Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY004Backbone")
RegNetY006Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY006Backbone")
RegNetY008Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY008Backbone")
RegNetY016Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY016Backbone")
RegNetY032Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY032Backbone")
RegNetY040Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY040Backbone")
RegNetY064Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY064Backbone")
RegNetY080Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY080Backbone")
RegNetY120Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY120Backbone")
RegNetY160Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY160Backbone")
RegNetY320Backbone.__doc__ = ALIAS_DOCSTRING.format(name="RegNetY320Backbone")
