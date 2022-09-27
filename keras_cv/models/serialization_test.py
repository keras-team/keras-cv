# Copyright 2022 The KerasCV Authors
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

import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models.csp_darknet import CSPDarkNet
from keras_cv.models.darknet import DarkNet21
from keras_cv.models.darknet import DarkNet53
from keras_cv.models.densenet import DenseNet121
from keras_cv.models.densenet import DenseNet169
from keras_cv.models.densenet import DenseNet201
from keras_cv.models.mlp_mixer import MLPMixerB16
from keras_cv.models.mlp_mixer import MLPMixerB32
from keras_cv.models.mlp_mixer import MLPMixerL16
from keras_cv.models.resnet_v1 import ResNet50
from keras_cv.models.resnet_v1 import ResNet101
from keras_cv.models.resnet_v1 import ResNet152
from keras_cv.models.resnet_v2 import ResNet50V2
from keras_cv.models.resnet_v2 import ResNet101V2
from keras_cv.models.resnet_v2 import ResNet152V2
from keras_cv.models.vgg19 import VGG19

def exhaustive_compare(obj1, obj2):
    classes_supporting_get_config = (
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
        return config_equals(obj1, obj2)

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
        return tf.keras.utils.serialize_keras_object(
            obj1
        ) == tf.keras.utils.serialize_keras_object(obj2)
    elif inspect.isfunction(obj1) and not inspect.isfunction(obj2):
        return tf.keras.utils.serialize_keras_object(obj1) == obj2
    elif inspect.isfunction(obj2) and not inspect.isfunction(obj1):
        return obj1 == tf.keras.utils.serialize_keras_object(obj2)

    # Lastly check for primitive datatypes and objects that don't need
    # additional preprocessing.
    else:
        return obj1 == obj2

def config_equals(config1, config2):
    if config1.keys() != config2.keys():
        return False
    
    for key in list(config1.keys()):
        v1, v2 = config1[key], config2[key]
        if not exhaustive_compare(v1, v2):
            return False
    
    return True

class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "CSP_DarkNet", 
            CSPDarkNet, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes": 4
            }
        ),

        (
            "DarkNet21", 
            DarkNet21, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "DarkNet53", 
            DarkNet53, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "DenseNet121", 
            DenseNet121, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "DenseNet169", 
            DenseNet169, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "DenseNet201", 
            DenseNet201, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "MLPMixerB16", 
            MLPMixerB16, 
            {   
                "input_shape": (224, 224, 3),
                "patch_size": (16, 16),
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "MLPMixerB32", 
            MLPMixerB32, 
            {
                "input_shape": (224, 224, 3),
                "patch_size": (16, 16),
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "MLPMixerL16", 
            MLPMixerL16, 
            {
                "input_shape": (224, 224, 3),
                "patch_size": (16, 16),
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "ResNet50", 
            ResNet50, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "ResNet101", 
            ResNet101, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4               
            }
        ),

        (
            "ResNet152", 
            ResNet152, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "ResNet50V2", 
            ResNet50V2, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "ResNet101V2", 
            ResNet101V2, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),

        (
            "ResNet152V2", 
            ResNet152V2, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        ),
        
        (
            "VGG19", 
            VGG19, 
            {
                "include_rescaling": True,
                "include_top": True,
                "classes" : 4
            }
        )
    )
    def test_model_serialization(self, model_cls, init_args):
        model = model_cls(**init_args)
        config = model.get_config()

        reconstructed_model = tf.keras.Model.from_config(config)
        self.assertTrue(
            config_equals(reconstructed_model.get_config(), config)
        )