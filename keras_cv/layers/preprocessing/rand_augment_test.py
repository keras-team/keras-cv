import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_cv import layers


class RandAugmentTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("0", 0),
        ("2", 2),
        ("5_5", 5.5),
        ("10", 10.0),
    )
    def test_runs_with_magnitude(self, magnitude):
        rand_augment = layers.RandAugment(magnitude=magnitude)
        xs = tf.ones((2, 512, 512, 3))
        ys = rand_augment(xs)
        self.assertEqual(ys.shape, (2, 512, 512, 3))
