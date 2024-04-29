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

import inspect

from tensorflow import keras

from keras_cv.src import core


def exhaustive_compare(obj1, obj2):
    """Exhaustively compared config of any two python
    or Keras objects recursively.

    If objects are python objects, a standard equality check is run.
    If the objects are Keras objects a `get_config()` call is made.
    The subsequent configs are then compared to determine if equality holds.

    Args:
        obj1: any object, can be a Keras object or python object.
        obj2: any object, can be a Keras object or python object.
    """

    classes_supporting_get_config = (
        core.FactorSampler,
        keras.layers.Layer,
        keras.losses.Loss,
    )

    # If both objects are either one of list or tuple then their individual
    # elements also must be checked exhaustively.
    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        # Length based checks.
        if len(obj1) == 0 and len(obj2) == 0:
            return True
        if len(obj1) != len(obj2):
            return False

        # Exhaustive check for all elements.
        for v1, v2 in list(zip(obj1, obj2)):
            return exhaustive_compare(v1, v2)

    # If the objects are dicts then we simply call the `config_equals` function
    # which supports dicts.
    elif isinstance(obj1, (dict)) and isinstance(obj2, (dict)):
        return config_equals(v1, v2)

    # If both objects are subclasses of Keras classes that support `get_config`
    # method, then we compare their individual attributes using `config_equals`.
    elif isinstance(obj1, classes_supporting_get_config) and isinstance(
        obj2, classes_supporting_get_config
    ):
        return config_equals(obj1.get_config(), obj2.get_config())

    # Following checks are if either of the objects are _functions_, not methods
    # or callables, since Layers and other unforeseen objects may also fit into
    # this category. Specifically for Keras activation functions.
    elif inspect.isfunction(obj1) and inspect.isfunction(obj2):
        return keras.utils.serialize_keras_object(
            obj1
        ) == keras.utils.serialize_keras_object(obj2)
    elif inspect.isfunction(obj1) and not inspect.isfunction(obj2):
        return keras.utils.serialize_keras_object(obj1) == obj2
    elif inspect.isfunction(obj2) and not inspect.isfunction(obj1):
        return obj1 == keras.utils.serialize_keras_object(obj2)

    # Lastly check for primitive datatypes and objects that don't need
    # additional preprocessing.
    else:
        return obj1 == obj2


def config_equals(config1, config2):
    # Both `config1` and `config2` are python dicts. So the first check is to
    # see if both of them have same keys.
    if config1.keys() != config2.keys():
        return False

    # Iterate over all keys of the configs and compare each entry exhaustively.
    for key in list(config1.keys()):
        v1, v2 = config1[key], config2[key]
        if not exhaustive_compare(v1, v2):
            return False
    return True
