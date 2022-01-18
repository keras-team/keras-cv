import tensorflow as tf
from keras_cv.layers.preprocessing.random_erase import RandomErase


NUM_CLASSES = 10


class RandomEraseTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, NUM_CLASSES)

        layer = RandomErase(1.0, patch_value=0.0, seed=1)
        xs, ys = layer(xs, ys)

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        # one hot labels
        self.assertEqual(ys.shape, [2, 10])
        self.assertEqual(len(ys != 0.0), 2)

    def test_random_erase_call_results(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((40, 40, 3)), tf.ones((40, 40, 3))], axis=0,), tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        patch_value = 0.0
        layer = RandomErase(1.0, patch_value=patch_value, seed=1)
        xs, ys = layer(xs, ys)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_cut_out_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))], axis=0,), tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        patch_value = 0.0
        layer = RandomErase(1.0, patch_value=patch_value, seed=1)
        xs, ys = layer(xs, ys)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0), tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        patch_value = 0.0
        layer = RandomErase(1.0, patch_value=patch_value, seed=1)

        @tf.function
        def augment(x, y):
            return layer(x, y)
        xs, ys = augment(xs, ys)

        # At least some pixels should be replaced in the RandomErase operation
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
