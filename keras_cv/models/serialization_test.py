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

def exhaustive_compare(v1, v2):
    pass

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

            }
        ),

        (
            "MLPMixerB32", 
            MLPMixerB32, 
            {

            }
        ),

        (
            "MLPMixerL16", 
            MLPMixerL16, 
            {

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
        self.assertAllInitParametersAreInConfig(model_cls, config)

        test_model = model
        test_model_config = test_model.get_config()

        reconstructed_model = tf.keras.Sequential().from_config(test_model_config)

        self.assertTrue(
            config_equals(reconstructed_model.get_config(), test_model_config)
        )


    def assertAllInitParametersAreInConfig(self, model_cls, config):
        exclude_name = ["args", "kwargs", "*"]
        parameter_names = {
            v for v in inspect.signature(model_cls).parameters.keys() 
            if v not in exclude_name
        }

        intersection_with_config = {
            v for v in config.keys() 
            if v in parameter_names
        }
        
        self.assertSetEqual(parameter_names, intersection_with_config)
        
