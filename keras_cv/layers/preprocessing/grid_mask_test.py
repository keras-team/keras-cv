import tensorflow as tf
from keras_cv.layers.preprocessing.grid_mask import GridMask


class GridMaskTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))

        layer = GridMask(
            ratio=0.1,
            gridmask_rotation_factor=(-0.2, 0.3)
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

        fill_value = 1
        layer = GridMask(
            ratio=1.0,
            gridmask_rotation_factor=(0.2, 0.3),
            fill_mode="constant",
            fill_value=fill_value
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_non_square_image(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((1024, 512, 1)), tf.ones((1024, 512, 1))],
                axis=0,
            ),
            tf.float32,
        )

        fill_value = 100
        layer = GridMask(
            ratio=0.6,
            gridmask_rotation_factor=0.3,
            fill_mode="constant",
            fill_value=fill_value
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0),
            tf.float32,
        )

        fill_value = 255
        layer = GridMask(
            ratio=0.4,
            gridmask_rotation_factor=0.5,
            fill_mode="constant",
            fill_value=fill_value
        )

        @tf.function
        def augment(x):
            return layer(x)

        xs = augment(xs)

        # Some pixels should be replaced with fill_value
        self.assertTrue(tf.math.reduce_any(xs[0] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == float(fill_value)))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_in_single_image(self):
        xs = tf.cast(
            tf.ones((512, 512, 1)), 
            tf.float32,
        )

        layer = GridMask(
            ratio=0.2,
            fill_mode="gaussian_noise",
        )
        xs = layer(xs)
        self.assertTrue(tf.math.reduce_any(xs == 0.0))
        self.assertTrue(tf.math.reduce_any(xs == 1.0))