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
import tensorflow as tf
from absl.testing import parameterized

from keras_cv import core
from keras_cv import metrics
from keras_cv.layers import preprocessing
from keras_cv.layers import regularization


def custom_compare(obj1, obj2):
    if isinstance(obj1, core.FactorSampler):
        return obj1.get_config() == obj2.get_config()
    else:
        return obj1 == obj2


def config_equals(config1, config2):
    for key in list(config1.keys()) + list(config2.keys()):
        v1, v2 = config1[key], config2[key]
        if not custom_compare(v1, v2):
            return False
    return True


class SerializationTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("COCORecall", metrics.COCORecall, {"class_ids": [0, 1, 2]}),
        (
            "COCOMeanAveragePrecision",
            metrics.COCOMeanAveragePrecision,
            {"class_ids": [0, 1, 2]},
        ),
    )
    def test_layer_serialization(self, metric_cls, init_args):
        metric = metric_cls(**init_args)
        metric_config = metric.get_config()
        reconstructed_metric = metric_cls.from_config(metric_config)

        self.assertTrue(
            config_equals(metric.get_config(), reconstructed_metric.get_config())
        )
