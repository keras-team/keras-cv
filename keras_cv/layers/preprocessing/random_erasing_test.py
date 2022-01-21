import tensorflow as tf

from keras_cv.layers.preprocessing import random_erasing

NUM_CLASSES = 10


class RandomErasingTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = random_erasing.RandomErasing(1.0, seed=1)
        xs = layer(xs)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_random_erasing_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 3)), tf.ones((40, 40, 3))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = random_erasing.RandomErasing(
            1.0, fill_mode="constant", fill_value=patch_value, seed=1
        )
        xs = layer(xs)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_random_erasing_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = random_erasing.RandomErasing(
            1.0, fill_mode="constant", fill_value=patch_value, seed=1
        )
        xs = layer(xs)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )

        patch_value = 0.0
        layer = random_erasing.RandomErasing(
            1.0, fill_mode="constant", fill_value=patch_value, seed=1
        )

        @tf.function
        def augment(x):
            return layer(x)

        xs = augment(xs)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
