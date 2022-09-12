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

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras_cv.models import resnet_v2

from .models_test import ModelsTest

MODEL_LIST = [
    (resnet_v2.ResNet50V2, 2048, {}, 0.892),
    (resnet_v2.ResNet101V2, 2048, {}, 0.873),
    (resnet_v2.ResNet152V2, 2048, {}, 0.852),
]


class ResNetV2Test(ModelsTest, tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(*MODEL_LIST)
    def test_application_base(self, app, last_dim, args, accuracy):
        super()._test_application_base(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_with_rescaling(self, app, last_dim, args, accuracy):
        super()._test_application_with_rescaling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_pooling(self, app, last_dim, args, accuracy):
        super()._test_application_pooling(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_application_variable_input_channels(self, app, last_dim, args, accuracy):
        super()._test_application_variable_input_channels(app, last_dim, args)

    @parameterized.parameters(*MODEL_LIST)
    def test_model_can_be_used_as_backbone(self, app, last_dim, args, accuracy):
        super()._test_model_can_be_used_as_backbone(app, last_dim, args)

    @pytest.mark.skipif(
        "CONVERGENCE" not in os.environ or os.environ["CONVERGENCE"] != "true",
        reason="Takes a long time to run, only runs when CONVERGENCE "
        "environment variable is set.  To run the test please run: \n"
        "`CONVERGENCE=true pytest keras_cv/",
    )
    @parameterized.parameters(*MODEL_LIST)
    def test_model_convergence(self, app, _, args, accuracy):
        super()._test_model_convergence(app, args, accuracy)


if __name__ == "__main__":
    tf.test.main()
