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
"""Base class for Backbone models."""

from keras_cv.src.api_export import keras_cv_export
from keras_cv.src.backend import keras
from keras_cv.src.utils.preset_utils import check_preset_class
from keras_cv.src.utils.preset_utils import load_from_preset
from keras_cv.src.utils.python_utils import classproperty
from keras_cv.src.utils.python_utils import format_docstring


@keras_cv_export("keras_cv.models.Backbone")
class Backbone(keras.Model):
    """Base class for Backbone models.

    Backbones are reusable layers of models trained on a standard task such as
    Imagenet classification that can be reused in other tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pyramid_level_inputs = {}
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )

    def __dir__(self):
        # Temporary fixes for weight saving. This mimics the following PR for
        # older version of Keras: https://github.com/keras-team/keras/pull/18982
        def filter_fn(attr):
            try:
                return id(getattr(self, attr)) not in self._functional_layer_ids
            except:
                return True

        return filter(filter_fn, super().__dir__())

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Backbone constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        """Dictionary of preset names and configs."""
        return {}

    @classproperty
    def presets_with_weights(cls):
        """Dictionary of preset names and configs that include weights."""
        return {}

    @classproperty
    def presets_without_weights(cls):
        """Dictionary of preset names and configs that don't include weights."""
        return {
            preset: cls.presets[preset]
            for preset in set(cls.presets) - set(cls.presets_with_weights)
        }

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        **kwargs,
    ):
        """Instantiate {{model_name}} model from preset config and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
                If looking for a preset with pretrained weights, choose one of
                "{{preset_with_weights_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `None`, which follows whether the preset has
                pretrained weights available.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_cv.models.{{model_name}}.from_preset(
            "{{example_preset_name}}",
        )

        # Load randomly initialized model from preset architecture with weights
        model = keras_cv.models.{{model_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False,
        ```
        """
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        check_preset_class(preset, cls)
        return load_from_preset(
            preset,
            load_weights=load_weights,
            config_overrides=kwargs,
        )

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to set up a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        if not cls.presets:
            cls.from_preset.__func__.__doc__ = """Not implemented.

            No presets available for this class.
            """

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)

    @property
    def pyramid_level_inputs(self):
        """Intermediate model outputs for feature extraction.

        Format is a dictionary with string as key and layer name as value.
        The string key represents the level of the feature output. A typical
        feature pyramid has five levels corresponding to scales "P3", "P4",
        "P5", "P6", "P7" in the backbone. Scale Pn represents a feature map 2^n
        times smaller in width and height than the input image.

        Example:
        ```python
        {
            'P3': 'v2_stack_1_block4_out',
            'P4': 'v2_stack_2_block6_out',
            'P5': 'v2_stack_3_block3_out',
        }
        ```
        """
        return self._pyramid_level_inputs

    @pyramid_level_inputs.setter
    def pyramid_level_inputs(self, value):
        self._pyramid_level_inputs = value
