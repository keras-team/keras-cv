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
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import _create_bounding_box_dataset


class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        tf.keras.backend.clear_session()

    # TODO(lukewood): configure for other coordinate systems.
    @pytest.mark.skipif(
        "INTEGRATION" not in os.environ or os.environ["INTEGRATION"] != "true",
        reason="Takes a long time to run, only runs when INTEGRATION "
        "environment variable is set.  To run the test please run: \n"
        "`INTEGRATION=true pytest "
        "keras_cv/models/object_detection/retina_net/retina_net_test.py -k "
        "test_fit_coco_metrics -s`",
    )
    def test_fit_coco_metrics(self):
        retina_net = keras_cv.models.RetinaNet(
            classes=2,
            bounding_box_format="xywh",
            backbone=keras_cv.models.ResNet50(
                include_top=False, include_rescaling=False
            ).as_backbone(),
        )
        # retina_net.backbone.trainable = False
        retina_net.compile(
            optimizer=optimizers.SGD(learning_rate=0.0075, momentum=0.9, global_clipnorm=10.0),
            classification_loss="focal",
            box_loss="smoothl1",
            metrics=[
                keras_cv.metrics.MeanBoxCountDelta(),
                keras_cv.metrics._BoxRecall(bounding_box_format="xywh", class_ids=[0]),
            ],
        )

        xs, ys = _create_bounding_box_dataset("xywh")
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.batch(xs.shape[0])
        retina_net.fit(ds.repeat(10), epochs=100)
        metrics = retina_net.evaluate(x=xs, y=ys, return_dict=True)
        self.assertAllGreater(metrics["mean_box_count_delta"], 0.0)
