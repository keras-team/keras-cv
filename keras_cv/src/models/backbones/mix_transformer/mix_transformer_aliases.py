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
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_backbone import (  # noqa: E501
    MiTBackbone,
)
from keras_cv.src.models.backbones.mix_transformer.mix_transformer_backbone_presets import (  # noqa: E501
    backbone_presets,
)
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """MiT model.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        include_rescaling: bool, whether to rescale the inputs. If set to
            True, inputs will be passed through a `Rescaling(scale=1 / 255)`
            layer. Defaults to True.
        input_shape: optional shape tuple, defaults to (None, None, 3).
        input_tensor: optional Keras tensor (i.e., output of `layers.Input()`)
            to use as image input for the model.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    model = {name}Backbone()
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.MiTB0Backbone")
class MiTB0Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "mit_b0_imagenet": copy.deepcopy(
                backbone_presets["mit_b0_imagenet"]
            ),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.MiTB1Backbone")
class MiTB1Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return {}


@keras_cv_export("keras_cv.models.MiTB2Backbone")
class MiTB2Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return {}


@keras_cv_export("keras_cv.models.MiTB3Backbone")
class MiTB3Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return {}


@keras_cv_export("keras_cv.models.MiTB4Backbone")
class MiTB4Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b4", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return {}


@keras_cv_export("keras_cv.models.MiTB5Backbone")
class MiTB5Backbone(MiTBackbone):
    def __new__(
        cls,
        include_rescaling=True,
        input_shape=(224, 224, 3),
        input_tensor=None,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "include_rescaling": include_rescaling,
                "input_shape": input_shape,
                "input_tensor": input_tensor,
            }
        )
        return MiTBackbone.from_preset("mit_b5", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations."""
        return {}


setattr(
    MiTB0Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB0"),
)

setattr(
    MiTB1Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB1"),
)

setattr(
    MiTB2Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB2"),
)

setattr(
    MiTB3Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB3"),
)

setattr(
    MiTB4Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB4"),
)

setattr(
    MiTB5Backbone,
    "__doc__",
    ALIAS_DOCSTRING.format(name="MiTB5"),
)
