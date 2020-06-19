import numpy as np
import tensorflow as tf
from kerascv.layers.losses.hard_neg_miner import HardNegativeMining


def test_more_negative_than_positive():
    classification_losses = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    positive_mask = tf.constant([[0, 0, 0], [0, 0, 1]])
    negative_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    hard_miner_layer = HardNegativeMining()
    losses = hard_miner_layer(classification_losses, positive_mask, negative_mask)
    # n_positives is 1, while n_negatives is 5, so picking the top 3, which is .3, .4, .5
    expected_out = np.asarray([0.3, 1.5]).astype(np.float32)
    np.testing.assert_allclose(expected_out, losses)


def test_less_negative_than_positive():
    classification_losses = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    positive_mask = tf.constant([[0, 1, 0], [0, 0, 1]])
    negative_mask = tf.constant([[0, 0, 1], [1, 0, 0]])
    hard_miner_layer = HardNegativeMining()
    losses = hard_miner_layer(classification_losses, positive_mask, negative_mask)
    # n_positives is 2, while n_negatives is 2, so pick all negative samples
    expected_out = np.asarray([0.5, 1.0])
    np.testing.assert_allclose(expected_out, losses)


def test_zero_negative_values():
    classification_losses = tf.constant([[0.0, 0.0, 0.3], [0.0, 0.5, 0.6]])
    positive_mask = tf.constant([[0, 0, 0], [0, 0, 1]])
    negative_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    hard_miner_layer = HardNegativeMining()
    losses = hard_miner_layer(classification_losses, positive_mask, negative_mask)
    # n_positives is 1, while n_negatives is 5, but only 2 of them are non-zero,
    # so picking the 2, which is .3, .5
    expected_out = np.asarray([0.3, 1.1]).astype(np.float32)
    np.testing.assert_allclose(expected_out, losses)


def test_no_negatives():
    classification_losses = tf.constant([[0.0, 0.0, 0.3], [0.0, 0.5, 0.6]])
    positive_mask = tf.constant([[0, 0, 0], [0, 0, 1]])
    negative_mask = tf.constant([[0, 0, 0], [0, 0, 0]])
    hard_miner_layer = HardNegativeMining()
    losses = hard_miner_layer(classification_losses, positive_mask, negative_mask)
    # n_positives is 1, while n_negatives is 0
    expected_out = np.asarray([0.0, 0.6]).astype(np.float32)
    np.testing.assert_allclose(expected_out, losses)


def test_min_negative_examples():
    classification_losses = tf.constant([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    positive_mask = tf.constant([[0, 0, 0], [0, 0, 1]])
    negative_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    hard_miner_layer = HardNegativeMining(
        negative_positive_ratio=2, minimum_negative_examples=4
    )
    losses = hard_miner_layer(classification_losses, positive_mask, negative_mask)
    # n_positives is 1, while n_negatives is 5, need at least 4 negative examples, 2., 3., 4., 5
    expected_out = np.asarray([0.5, 1.5])
    np.testing.assert_allclose(expected_out, losses)


def test_config_with_custom_name():
    layer = HardNegativeMining(name="hard_example_miner")
    config = layer.get_config()
    layer_1 = HardNegativeMining.from_config(config)
    np.testing.assert_equal(layer_1.name, layer.name)
