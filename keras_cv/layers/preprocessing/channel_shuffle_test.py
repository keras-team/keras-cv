import tensorflow as tf
from keras_cv.layers.preprocessing.channel_shuffle import ChannelShuffle


class GridMaskTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = ChannelShuffle()
        xs = layer(xs, training=True)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_gridmask_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [3 * tf.ones((40, 40, 1)), 2*tf.ones((40, 40, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle()
        xs = layer(xs, training=True)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == 3.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 1)), tf.ones((1024, 512, 1))],
                axis=0,
            ),
            dtype=tf.float32,
        )

        layer = ChannelShuffle()
        xs = layer(xs, training=True)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            dtype=tf.float32,
        )

        layer = ChannelShuffle()

        @tf.function
        def augment(x):
            return layer(x, training=True)

        xs = augment(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 1)), 
            dtype=tf.float32,
        )

        layer = ChannelShuffle()
        xs = layer(xs, training=True)
        self.assertTrue(tf.math.reduce_any(xs == 1.0))