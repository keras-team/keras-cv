
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_cv import bounding_box


from keras import Model
from keras import backend
from keras import layers
from keras import metrics
from keras.testing_infra import test_combinations


class IoUTest(tf.test.TestCase):
    def test_config(self):
        obj = metrics.IoU(
            num_classes=2, target_class_ids=[1, 0], name="iou_class_1_0"
        )
        self.assertEqual(obj.name, "iou_class_1_0")
        self.assertEqual(obj.num_classes, 2)
        self.assertEqual(obj.target_class_ids, [1, 0])

        obj2 = metrics.IoU.from_config(obj.get_config())
        self.assertEqual(obj2.name, "iou_class_1_0")
        self.assertEqual(obj2.num_classes, 2)
        self.assertEqual(obj2.target_class_ids, [1, 0])

    def test_unweighted(self):
        y_pred = [0, 1, 0, 1]
        y_true = [0, 0, 1, 1]

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred)

        # cm = [[1, 1],
        #       [1, 1]]
        # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_weighted(self):
        y_pred = tf.constant([0, 1, 0, 1], dtype=tf.float32)
        y_true = tf.constant([0, 0, 1, 1])
        sample_weight = tf.constant([0.2, 0.3, 0.4, 0.1])

        obj = metrics.IoU(num_classes=2, target_class_ids=[1, 0])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.1 / (0.4 + 0.5 - 0.1) + 0.2 / (0.6 + 0.5 - 0.2)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_multi_dim_input(self):
        y_pred = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)
        y_true = tf.constant([[0, 0], [1, 1]])
        sample_weight = tf.constant([[0.2, 0.3], [0.4, 0.1]])

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))

        result = obj(y_true, y_pred, sample_weight=sample_weight)

        # cm = [[0.2, 0.3],
        #       [0.4, 0.1]]
        # sum_row = [0.6, 0.4], sum_col = [0.5, 0.5], true_positives = [0.2,
        # 0.1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (
            0.2 / (0.6 + 0.5 - 0.2) + 0.1 / (0.4 + 0.5 - 0.1)
        ) / 2
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)

    def test_zero_valid_entries(self):
        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        self.assertAllClose(self.evaluate(obj.result()), 0, atol=1e-3)

    def test_zero_and_non_zero_entries(self):
        y_pred = tf.constant([1], dtype=tf.float32)
        y_true = tf.constant([1])

        obj = metrics.IoU(num_classes=2, target_class_ids=[0, 1])
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        result = obj(y_true, y_pred)

        # cm = [[0, 0],
        #       [0, 1]]
        # sum_row = [0, 1], sum_col = [0, 1], true_positives = [0, 1]
        # iou = true_positives / (sum_row + sum_col - true_positives))
        expected_result = (1 / (1 + 1 - 1)) / 1
        self.assertAllClose(self.evaluate(result), expected_result, atol=1e-3)