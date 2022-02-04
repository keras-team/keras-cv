import tensorflow as tf
from keras_cv.layers.preprocessing.grid_mask import GridMask


class GridMaskTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = GridMask(
            ratio=0.6, 
            rate=1.0
        )
        xs = layer(xs)

        self.assertEqual(xs.shape, [2, 512, 512, 3])

    def test_gridmask_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )

        layer = GridMask(
            ratio=0.6,
            rate=1.0,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_gridmask_call_tiny_image(self):
        img_shape = (4, 4, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        layer = GridMask(
            ratio=0.6, 
            rate=1.0
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 1)), tf.ones((1024, 512, 1))],
                axis=0,
            ),
            tf.float32,
        )

        layer = GridMask(
            ratio=0.6,
            rate=1.0,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )

        layer = GridMask(
            ratio=0.6,
            rate=1.0,
        )

        @tf.function
        def augment(x):
            return layer(x)

        xs = augment(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
