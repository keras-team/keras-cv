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
"""Base class for Task models."""

import os

from keras_cv.api_export import keras_cv_export
from keras_cv.backend import keras
from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.python_utils import format_docstring


@keras_cv_export("keras_cv.models.Task")
class Task(keras.Model):
    """Base class for Task models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._backbone = None

    @property
    def backbone(self):
        """A `keras.Model` instance providing the backbone submodel."""
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        self._backbone = value

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Task constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
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

    @classproperty
    def backbone_presets(cls):
        """Dictionary of preset names and configs for compatible backbones."""
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=None,
        input_shape=None,
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
            input_shape : input shape that will be passed to backbone
                initialization, Defaults to `None`, which will not be passed to
                the backbone.

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

        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if load_weights and preset not in cls.presets_with_weights:
            raise ValueError(
                f"""Pretrained weights not available for preset "{preset}". """
                "Set `load_weights=False` to use this preset or choose one of "
                "the following presets with weights:"
                f""" "{'", "'.join(cls.presets_with_weights)}"."""
            )

        metadata = cls.presets[preset]
        # Check if preset is backbone-only model
        if preset in cls.backbone_presets:
            backbone_cls = keras.saving.get_registered_object(
                metadata["class_name"]
            )
            backbone = backbone_cls.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        config = metadata["config"]
        if input_shape is not None:
            config["backbone"]["config"]["input_shape"] = input_shape
        model = cls.from_config({**config, **kwargs})

        if preset not in cls.presets_with_weights or load_weights is False:
            return model

        local_weights_path = "model.h5"
        if metadata["weights_url"].endswith(".weights.h5"):
            local_weights_path = "model.weights.h5"

        weights = keras.utils.get_file(
            local_weights_path,
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model

    @property
    def layers(self):
        # Some of our task models don't use the Backbone directly, but create
        # a feature extractor from it. In these cases, we don't want to count
        # the `backbone` as a layer, because it will be included in the model
        # summary and saves weights despite not being part of the model graph.
        layers = super().layers
        if hasattr(self, "backbone") and self.backbone in layers:
            # We know that the backbone is not part of the graph if it has no
            # inbound nodes.
            if len(self.backbone._inbound_nodes) == 0:
                layers.remove(self.backbone)
        return layers

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["backbone"]:
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

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
            cls.from_preset.__func__.__doc__ = Task.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets_with_weights), ""),
                preset_names='", "'.join(cls.presets),
                preset_with_weights_names='", "'.join(cls.presets_with_weights),
            )(cls.from_preset.__func__)
