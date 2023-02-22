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

import os

import tensorflow as tf
from tensorflow import keras

from keras_cv.utils.python_utils import classproperty
from keras_cv.utils.python_utils import format_docstring


class Backbone(keras.Model):
    """Base class for Backbone models."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate {{model_name}} model from preset architecture and weights.
        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.
        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_cv.models.{{model_name}}.from_preset(
            "{{example_preset_name}}"
        )
        # Load randomly initialized model from preset architecture
        model = keras_cv.models.{{model_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
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
        metadata = cls.presets[preset]

        if load_weights and "weights_url" not in metadata:
            raise ValueError(
                f"""Pretrained weights not available for preset "{preset}". """
                "Set `load_weights=False` to use this preset."
            )

        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have an distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)

    @property
    def backbone_level_outputs(self):
        """Backbone outputs at each resolution level for transfer learning."""
        return None

    @backbone_level_outputs.setter
    def backbone_level_outputs(self, value):
        self._backbone_level_outputs = value

    def extract_features(self, min_level=None, max_level=None):
        """Convert the application model into a model backbone for other tasks.

        The backbone model will usually take same inputs as the original
        application model, but produce multiple outputs, one for each feature
        level. Those outputs can be feed to network downstream, like FPN and RPN.

        The output of the backbone model will be a dict with int as key and
        tensor as value. The int key represent the level of the feature output.
        A typical feature pyramid has five levels corresponding to scales P3,
        P4, P5, P6, P7 in the backbone. Scale Pn represents a feature map 2n
        times smaller in width and height than the input image.

        Args:
            min_level: optional int, the lowest level of feature to be included
                in the output. Default to model's lowest feature level (based on
                the model structure).
            max_level: optional int, the highest level of feature to be included
                in the output. Default to model's highest feature level (based
                on the model structure).

        Returns:
            a `tf.keras.Model` which has dict as outputs.
        Raises:
            ValueError: When the model is lack of information for feature level,
            and can't be converted to backbone model, or the min_level/max_level
            param is out of range based on the model structure.
        """
        if self._backbone_level_outputs is not None:
            backbone_level_outputs = self._backbone_level_outputs
            model_levels = list(sorted(backbone_level_outputs.keys()))
            if min_level is not None:
                if min_level < model_levels[0]:
                    raise ValueError(
                        f"The min_level provided: {min_level} should be in "
                        f"the range of {model_levels}"
                    )
            else:
                min_level = model_levels[0]

            if max_level is not None:
                if max_level > model_levels[-1]:
                    raise ValueError(
                        f"The max_level provided: {max_level} should be in "
                        f"the range of {model_levels}"
                    )
            else:
                max_level = model_levels[-1]

            outputs = {}
            for level in range(min_level, max_level + 1):
                outputs[level] = backbone_level_outputs[level]

            return tf.keras.Model(inputs=self.inputs, outputs=outputs)

        else:
            raise ValueError(
                "The current model doesn't have any feature level "
                "information so extraction not possible."
            )
