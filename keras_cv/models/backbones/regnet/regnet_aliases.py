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
from keras_cv.models.backbones.regnet.regnet_backbone import (  # noqa: E501
    RegNetBackbone,
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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

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
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return {}


setattr(
    RegNetX002Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX002Backbone"),
)
setattr(
    RegNetX004Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX004Backbone"),
)
setattr(
    RegNetX006Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX006Backbone"),
)
setattr(
    RegNetX008Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX008Backbone"),
)
setattr(
    RegNetX016Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX016Backbone"),
)
setattr(
    RegNetX032Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX032Backbone"),
)
setattr(
    RegNetX040Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX040Backbone"),
)
setattr(
    RegNetX064Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX064Backbone"),
)
setattr(
    RegNetX080Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX080Backbone"),
)
setattr(
    RegNetX120Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX120Backbone"),
)
setattr(
    RegNetX160Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX160Backbone"),
)
setattr(
    RegNetX320Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetX320Backbone"),
)

setattr(
    RegNetY002Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY002Backbone"),
)
setattr(
    RegNetY004Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY004Backbone"),
)
setattr(
    RegNetY006Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY006Backbone"),
)
setattr(
    RegNetY008Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY008Backbone"),
)
setattr(
    RegNetY016Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY016Backbone"),
)
setattr(
    RegNetY032Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY032Backbone"),
)
setattr(
    RegNetY040Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY040Backbone"),
)
setattr(
    RegNetY064Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY064Backbone"),
)
setattr(
    RegNetY080Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY080Backbone"),
)
setattr(
    RegNetY120Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY120Backbone"),
)
setattr(
    RegNetY160Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY160Backbone"),
)
setattr(
    RegNetY320Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="RegNetY320Backbone"),
)
