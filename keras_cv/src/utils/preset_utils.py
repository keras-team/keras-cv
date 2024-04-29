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

import datetime
import inspect
import json
import os

from keras_cv.src.backend import keras

try:
    import kagglehub
except ImportError:
    kagglehub = None

KAGGLE_PREFIX = "kaggle://"
GS_PREFIX = "gs://"


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    if not isinstance(preset, str):
        raise ValueError(
            f"A preset identifier must be a string. Received: preset={preset}"
        )
    if preset.startswith(KAGGLE_PREFIX):
        if kagglehub is None:
            raise ImportError(
                "`from_preset()` requires the `kagglehub` package. "
                "Please install with `pip install kagglehub`."
            )
        # Insert the kaggle framework into the handle.
        kaggle_handle = preset.removeprefix(KAGGLE_PREFIX)
        num_segments = len(kaggle_handle.split("/"))
        if num_segments not in (4, 5):
            raise ValueError(
                "Unexpected kaggle preset handle. Kaggle model handles "
                "should have the form "
                "kaggle://{org}/{model}/keras/{variant}[/{version}]. "
                "For example, "
                "'kaggle://keras/retinanet/keras/retinanet_base_en'. "
                f"Received: preset={preset}"
            )
        return kagglehub.model_download(kaggle_handle, path)
    elif preset.startswith(GS_PREFIX):
        url = os.path.join(preset, path)
        url = url.replace(GS_PREFIX, "https://storage.googleapis.com/")
        subdir = preset.replace(GS_PREFIX, "gs_")
        subdir = subdir.replace("/", "_").replace("-", "_")
        filename = os.path.basename(path)
        subdir = os.path.join(subdir, os.path.dirname(path))
        return keras.utils.get_file(
            filename,
            url,
            cache_subdir=os.path.join("models", subdir),
        )
    elif os.path.exists(preset):
        # Assume a local filepath.
        return os.path.join(preset, path)
    else:
        raise ValueError(
            "Unknown preset identifier. A preset must be a one of:\n"
            "1) a built in preset identifier like `'mobilenet_v3_small'`\n"
            "2) a Kaggle Models handle like `'kaggle://keras/mobilenetv3/keras/mobilenet_v3_small'`\n"  # noqa: E501
            "3) a path to a local preset directory like `'./mobilenet_v3_small`\n"  # noqa: E501
            "Use `print(cls.presets.keys())` to view all built-in presets for "
            "API symbol `cls`.\n"
            f"Received: preset='{preset}'"
        )


def recursive_pop(config, key):
    """Remove a key from a nested config object"""
    config.pop(key, None)
    for value in config.values():
        if isinstance(value, dict):
            recursive_pop(value, key)
        if isinstance(value, list):
            for v in value:
                if isinstance(v, dict):
                    recursive_pop(v, key)


def save_to_preset(
    layer,
    preset,
    save_weights=True,
    config_filename="config.json",
    weights_filename="model.weights.h5",
):
    """Save a KerasCV layer to a preset directory."""
    os.makedirs(preset, exist_ok=True)

    # Optionally save weights.
    save_weights = save_weights and hasattr(layer, "save_weights")
    if save_weights:
        weights_path = os.path.join(preset, weights_filename)
        layer.save_weights(weights_path)

    # Save a serialized Keras object.
    config_path = os.path.join(preset, config_filename)
    config = keras.saving.serialize_keras_object(layer)
    # Include references to weights.
    config["weights"] = weights_filename if save_weights else None
    recursive_pop(config, "compile_config")
    recursive_pop(config, "build_config")
    with open(config_path, "w") as config_file:
        config_file.write(json.dumps(config, indent=4))

    from keras_cv.src import __version__ as keras_cv_version

    keras_version = keras.version() if hasattr(keras, "version") else None

    # Save any associated metadata.
    if config_filename == "config.json":
        metadata = {
            "keras_version": keras_version,
            "keras_cv_version": keras_cv_version,
            "parameter_count": layer.count_params(),
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
        metadata_path = os.path.join(preset, "metadata.json")
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))


def load_from_preset(
    preset,
    load_weights=None,
    input_shape=None,
    config_file="config.json",
    config_overrides={},
):
    """Load a KerasCV layer to a preset directory."""
    # Load a serialized Keras object.
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["config"] = {**config["config"], **config_overrides}
    layer = keras.saving.deserialize_keras_object(config)
    if input_shape is not None:
        layer.build(input_shape)

    # Check load_weights flag does not violate preset config.
    if load_weights is True and config["weights"] is None:
        raise ValueError(
            f"The specified preset `{preset}` does not include weights. "
            "Please remove the `load_weights` flag when calling "
            "`from_preset()` on this preset."
        )
    # Default to loading weights if available.
    if load_weights is not False and config["weights"] is not None:
        weights_path = get_file(preset, config["weights"])
        if hasattr(layer, "_layer_checkpoint_dependencies"):
            legacy_load_weights(layer, weights_path)
        else:
            layer.load_weights(weights_path)

    return layer


def check_preset_class(
    preset,
    classes,
    config_file="config.json",
):
    """Validate a preset is being loaded on the correct class."""
    config_path = get_file(preset, config_file)
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except:
        raise ValueError(
            f"The specified preset  `{preset}` is unknown. "
            "Please check documentation to ensure the correct preset "
            "handle is being used."
        )
    cls = keras.saving.get_registered_object(config["registered_name"])
    if not isinstance(classes, (tuple, list)):
        classes = (classes,)

    # Subclass checking and alias checking
    if not any(issubclass(cls, obj) for obj in classes) and not any(
        issubclass(alias, cls) for alias in classes
    ):
        raise ValueError(
            f"Unexpected class in preset `'{preset}'`. "
            "When calling `from_preset()` on a class object, the preset class "
            f"much match allowed classes. Allowed classes are `{classes}`. "
            f"Received: `{cls}`."
        )
    return cls


def legacy_load_weights(layer, weights_path):
    # Hacky fix for TensorFlow 2.13 and 2.14 when loading a `.weights.h5` file.
    # We find the `Functional` class, and temporarily remove the
    # `_layer_checkpoint_dependencies` property, which on older version of
    # TensorFlow complete broke the variable paths for functional models.
    functional_cls = None
    for cls in inspect.getmro(layer.__class__):
        if cls.__name__ == "Functional":
            functional_cls = cls
    property = functional_cls._layer_checkpoint_dependencies
    functional_cls._layer_checkpoint_dependencies = {}
    layer.load_weights(weights_path)
    functional_cls._layer_checkpoint_dependencies = property
