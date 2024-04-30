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
from keras_cv.src.models.segmentation.segformer.segformer import SegFormer
from keras_cv.src.models.segmentation.segformer.segformer_presets import presets
from keras_cv.src.utils.python_utils import classproperty

ALIAS_DOCSTRING = """SegFormer model.

    For transfer learning use cases, make sure to read the
    [guide to transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/).

    Args:
        backbone: a KerasCV backbone for feature extraction.
        num_classes: the number of classes for segmentation, including the background class.

    Example:
    ```python
    input_data = tf.ones(shape=(8, 224, 224, 3))

    # Randomly initialized backbone
    backbone = keras_cv.models.MiTBackbone.from_preset("mit_b0_imagenet")
    segformer = keras_cv.models.SegFormer(backbone=backbone, num_classes=19)
    output = model(input_data)
    ```
"""  # noqa: E501


@keras_cv_export("keras_cv.models.SegFormerB0")
class SegFormerB0(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b0", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b0": copy.deepcopy(presets["segformer_b0"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.SegFormerB1")
class SegFormerB1(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b1", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b1": copy.deepcopy(presets["segformer_b1"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.SegFormerB2")
class SegFormerB2(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b2", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b2": copy.deepcopy(presets["segformer_b2"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.SegFormerB3")
class SegFormerB3(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b3", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b3": copy.deepcopy(presets["segformer_b3"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.SegFormerB4")
class SegFormerB4(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b4", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b4": copy.deepcopy(presets["segformer_b4"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


@keras_cv_export("keras_cv.models.SegFormerB5")
class SegFormerB5(SegFormer):
    def __new__(
        cls,
        num_classes,
        **kwargs,
    ):
        # Pack args in kwargs
        kwargs.update(
            {
                "num_classes": num_classes,
            }
        )
        return SegFormer.from_preset("segformer_b5", **kwargs)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configurations."""
        return {
            "segformer_b5": copy.deepcopy(presets["segformer_b5"]),
        }

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configurations that include
        weights."""
        return cls.presets


setattr(
    SegFormerB0,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB0"),
)

setattr(
    SegFormerB1,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB1"),
)

setattr(
    SegFormerB2,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB2"),
)

setattr(
    SegFormerB3,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB3"),
)

setattr(
    SegFormerB4,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB4"),
)

setattr(
    SegFormerB5,
    "__doc__",
    ALIAS_DOCSTRING.format(name="SegFormerB5"),
)
