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

import tensorflow as tf
from absl.testing import parameterized

from keras_cv import losses as cv_losses
from keras_cv.utils import test_utils


class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        (
            "FocalLoss",
            cv_losses.FocalLoss,
            {"alpha": 0.25, "gamma": 2, "from_logits": True},
        ),
        ("GIoULoss", cv_losses.GIoULoss, {"bounding_box_format": "xywh"}),
        (
            "BinaryPenaltyReducedFocalCrossEntropy",
            cv_losses.BinaryPenaltyReducedFocalCrossEntropy,
            {},
        ),
        ("SimCLRLoss", cv_losses.SimCLRLoss, {"temperature": 0.5}),
        ("SmoothL1Loss", cv_losses.SmoothL1Loss, {}),
    )
    def test_loss_serialization(self, loss_cls, init_args):
        loss = loss_cls(**init_args)
        config = loss.get_config()
        self.assertAllInitParametersAreInConfig(loss_cls, config)

        reconstructed_loss = loss_cls.from_config(config)

        self.assertTrue(
            test_utils.config_equals(
                loss.get_config(), reconstructed_loss.get_config()
            )
        )

    def assertAllInitParametersAreInConfig(self, loss_cls, config):
        excluded_name = ["args", "kwargs", "*"]
        parameter_names = {
            v
            for v in inspect.signature(loss_cls).parameters.keys()
            if v not in excluded_name
        }

        intersection_with_config = {
            v for v in config.keys() if v in parameter_names
        }

        self.assertSetEqual(parameter_names, intersection_with_config)
