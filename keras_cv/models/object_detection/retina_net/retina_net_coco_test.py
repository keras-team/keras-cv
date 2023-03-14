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

import pytest
import tensorflow as tf
from tensorflow.keras import optimizers

import keras_cv
from keras_cv.models.object_detection.__test_utils__ import (
    _create_bounding_box_dataset,
)


class RetinaNetTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def cleanup_global_session(self):
        # Code before yield runs before the test
        tf.config.set_soft_device_placement(False)
        yield
        # Reset soft device placement to not interfere with other unit test files
        tf.config.set_soft_device_placement(True)
        tf.keras.backend.clear_session()

    def test_fit_coco_metrics(self):
        retina_net = keras_cv.models.RetinaNet(
            num_classes=2,
            bounding_box_format="xywh",
        )
        retina_net.prediction_decoder = (
            keras_cv.layers.MultiClassNonMaxSuppression(
                bounding_box_format="xywh",
                confidence_threshold=0.9,
                from_logits=True,
            )
        )

        # BatchNormalization makes it really hard to overfit a miniature dataset
        for layer in retina_net.backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        retina_net.compile(
            optimizer=optimizers.SGD(
                learning_rate=0.0075, global_clipnorm=10.0
            ),
            classification_loss="focal",
            box_loss="smoothl1",
            metrics=[
                keras_cv.metrics._BoxRecall(
                    bounding_box_format="xywh",
                    iou_thresholds=[0.5],
                    class_ids=[0],
                ),
            ],
        )

        class StopWhenMetricAboveThreshold(tf.keras.callbacks.Callback):
            def __init__(self, metric, threshold):
                self.metric = metric
                self.threshold = threshold

            def on_epoch_end(self, epoch, logs):
                if logs[self.metric] > self.threshold:
                    self.model.stop_training = True

        xs, ys = _create_bounding_box_dataset("xywh")
        dataset = tf.data.Dataset.from_tensor_slices((xs, ys))
        dataset = dataset.batch(xs.shape[0])

        recall_target_threshold = 0.75

        retina_net.fit(
            dataset,
            validation_data=dataset,
            epochs=50,
            callbacks=[
                StopWhenMetricAboveThreshold(
                    "val_private__box_recall", recall_target_threshold
                )
            ],
        )
        metrics = retina_net.evaluate(x=xs, y=ys, return_dict=True)
        self.assertAllGreater(
            metrics["private__box_recall"], recall_target_threshold
        )
